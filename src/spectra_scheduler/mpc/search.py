"""Single-root and batched finite-horizon Monte Carlo tree search."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT_GAMMA, DEFAULT_MCTS_SIMS, MAX_BANDS
from .model import NeuralMPCModel


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = ("prior", "visit_count", "value_sum", "reward", "latent_state", "children")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0
        self.latent_state: torch.Tensor | None = None
        self.children: dict[int, MCTSNode] = {}

    @property
    def expanded(self) -> bool:
        return self.latent_state is not None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Monte Carlo Tree Search with PUCT selection and neural rollouts.

    At each planning step the search builds a tree of imagined futures
    using the learned dynamics model.  Actions are selected at internal
    nodes via the PUCT formula (balancing exploitation of Q-values with
    exploration proportional to the policy prior).  Leaf nodes are
    evaluated by the prediction network's value head.

    After all simulations the action at the root with the highest visit
    count is selected — this is the standard MuZero action-selection
    strategy.
    """

    def __init__(
        self,
        model: NeuralMPCModel,
        num_bands: int,
        *,
        num_simulations: int = DEFAULT_MCTS_SIMS,
        c_puct: float = 1.5,
        gamma: float = DEFAULT_GAMMA,
        max_depth: int = 5,
        device: torch.device | None = None,
    ) -> None:
        if num_simulations < 1 or max_depth < 1 or not 1 <= num_bands <= MAX_BANDS:
            raise ValueError("positive search budget/depth and valid band count required")
        self.model = model
        self.num_bands = num_bands
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.max_depth = max_depth
        self.device = device or torch.device("cpu")

    @torch.inference_mode()
    def search(self, root_state: torch.Tensor) -> tuple[int, np.ndarray]:
        """Run MCTS and return *(best_action, visit_distribution)*.

        Args:
            root_state: 1-D tensor of shape ``(hidden_size,)``.

        Returns:
            best_action: integer band index.
            visit_dist: array of shape ``(num_bands,)`` with normalised
                visit fractions (useful as a training target).
        """
        # Expand root -------------------------------------------------------
        root = MCTSNode(prior=1.0)
        root.latent_state = root_state

        policy_logits, root_value = self.model.predict(root_state)
        priors = F.softmax(policy_logits[0, : self.num_bands], dim=-1)
        for a in range(self.num_bands):
            root.children[a] = MCTSNode(prior=priors[a].item())
        root.visit_count = 1
        root.value_sum = root_value.item()

        # Simulations -------------------------------------------------------
        for _ in range(self.num_simulations):
            node = root
            search_path: list[MCTSNode] = [node]
            actions_taken: list[int] = []

            # SELECT — descend using PUCT until we hit an unexpanded leaf
            while node.expanded and node.children and len(actions_taken) < self.max_depth:
                action, child = self._select_child(node)
                actions_taken.append(action)
                search_path.append(child)
                node = child

            # EXPAND — use dynamics to materialise the leaf
            parent = search_path[-2] if len(search_path) >= 2 else root
            leaf = search_path[-1]
            action = actions_taken[-1] if actions_taken else 0

            if parent.latent_state is not None and not leaf.expanded:
                act_t = torch.tensor([action], dtype=torch.long, device=self.device)
                ns, rew = self.model.dynamics(parent.latent_state.unsqueeze(0), act_t)
                leaf.latent_state = ns.squeeze(0)
                leaf.reward = float(np.clip(rew.item(), -0.25, 1.0))

                child_logits, leaf_value = self.model.predict(leaf.latent_state)
                child_priors = F.softmax(child_logits[0, : self.num_bands], dim=-1)
                for a in range(self.num_bands):
                    if a not in leaf.children:
                        leaf.children[a] = MCTSNode(prior=child_priors[a].item())
                value = leaf_value.item()
            else:
                _, estimate = self.model.predict(leaf.latent_state)
                value = estimate.item()

            # BACKPROPAGATE
            self._backprop(search_path, value)

        # Action selection by visit count ------------------------------------
        visits = np.zeros(self.num_bands)
        for a, ch in root.children.items():
            visits[a] = ch.visit_count
        best = int(np.argmax(visits))
        total = visits.sum()
        dist = visits / total if total > 0 else np.ones(self.num_bands) / self.num_bands
        return best, dist

    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        total = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total + 1)

        best_score = -math.inf
        best_action = 0
        best_child = next(iter(node.children.values()))

        for action, child in node.children.items():
            q = child.reward + self.gamma * child.value if child.visit_count > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backprop(self, path: list[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = node.reward + self.gamma * value


@torch.inference_mode()
def search_batch(model, states, remaining, cfg, rng, explore=False):
    """One leaf per world per inference batch; finite-horizon discounted backups."""
    if len(states) != len(remaining) or not len(states) or min(remaining) < 1:
        raise ValueError("one positive remaining horizon per root is required")
    helper = MCTS(
        model, MAX_BANDS, num_simulations=cfg.simulations, gamma=cfg.gamma, max_depth=cfg.depth
    )
    logits, _ = model.predict(states)
    priors = logits.softmax(-1).cpu().numpy()
    roots = []
    for i, state in enumerate(states):
        root = MCTSNode(1.0)
        root.latent_state = state
        p = priors[i]
        if explore:
            p = 0.75 * p + 0.25 * rng.dirichlet(np.full(MAX_BANDS, 0.3))
        root.children = {a: MCTSNode(float(p[a])) for a in range(MAX_BANDS)}
        roots.append(root)
    for _ in range(cfg.simulations):
        paths, pending, actions, parent_states = [], [], [], []
        for i, root in enumerate(roots):
            node, path = root, [root]
            limit = min(cfg.depth, int(remaining[i]))
            while node.expanded and node.children and len(path) - 1 < limit:
                action, node = helper._select_child(node)
                path.append(node)
            paths.append(path)
            if not node.expanded:
                pending.append(i)
                actions.append(action)
                parent_states.append(path[-2].latent_state)
        if pending:
            ns, rewards = model.dynamics(
                torch.stack(parent_states), torch.tensor(actions, device=states.device)
            )
            child_logits, estimates = model.predict(ns)
            probabilities = child_logits.softmax(-1).cpu().numpy()
            for j, i in enumerate(pending):
                leaf = paths[i][-1]
                leaf.latent_state = ns[j]
                leaf.reward = float(rewards[j].clamp(-0.25, 1))
                leaf.children = {a: MCTSNode(float(probabilities[j, a])) for a in range(MAX_BANDS)}
                # Store the network estimate separately from backed-up values.
            estimates = {i: float(estimates[j]) for j, i in enumerate(pending)}
        else:
            estimates = {}
        for i, path in enumerate(paths):
            steps_left = int(remaining[i]) - len(path) + 1
            if steps_left <= 0:
                value = 0.0
            elif i in estimates:
                value = estimates[i]
            else:
                _, v = model.predict(path[-1].latent_state)
                value = float(v.item())
            bound = (1 - cfg.gamma ** max(0, steps_left)) / (1 - cfg.gamma)
            value = float(np.clip(value, -0.25 * bound, bound))
            helper._backprop(path, value)
    visits = np.array(
        [[c.visit_count for c in r.children.values()] for r in roots], dtype=np.float32
    )
    visits /= visits.sum(axis=1, keepdims=True)
    # Root value is the mean backed-up return, not a privileged simulator target.
    return visits, np.array([r.value for r in roots], dtype=np.float32)
