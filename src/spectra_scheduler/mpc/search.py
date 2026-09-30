"""Single-root and batched finite-horizon Monte Carlo tree search."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT_GAMMA, DEFAULT_MCTS_SIMS, DWELL_STEPS, MAX_BANDS
from .model import NeuralMPCModel


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "reward",
        "latent_state",
        "children",
        "estimate",
        "duration",
        "elapsed",
        "band",
    )

    def __init__(
        self, prior: float = 0.0, *, duration: int = 1, elapsed: int = 0, band: int = -1
    ) -> None:
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0
        self.latent_state: torch.Tensor | None = None
        self.children: dict[int, MCTSNode] = {}
        self.estimate: float | None = None
        self.duration = duration
        self.elapsed = elapsed
        self.band = band

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

    def _select_child(self, node: MCTSNode, bounds=None) -> tuple[int, MCTSNode]:
        total = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total + 1)

        best_score = -math.inf
        best_action = 0
        best_child = next(iter(node.children.values()))

        for action, child in node.children.items():
            q = (
                child.reward + self.gamma**child.duration * child.value
                if child.visit_count > 0
                else 0.0
            )
            if bounds is not None:
                low, high = bounds
                q = (
                    min(1.0, max(0.0, (q - low) / (high - low)))
                    if child.visit_count and high > low + 1e-8
                    else 0.0
                )
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    @staticmethod
    def _select_gumbel_child(node: MCTSNode, gamma: float, scale: float):
        """Choose a non-root action from a completed-Q improved policy."""
        actions = list(node.children)
        children = [node.children[action] for action in actions]
        baseline = 0.0 if node.estimate is None else node.estimate
        q = np.asarray(
            [
                child.reward + gamma**child.duration * child.value
                if child.visit_count else baseline
                for child in children
            ],
            dtype=np.float64,
        )
        span = q.max() - q.min()
        normalized = (q - q.min()) / span if span > 1e-8 else np.zeros_like(q)
        logits = np.log(np.maximum([child.prior for child in children], 1e-12))
        logits += scale * normalized
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        counts = np.asarray([child.visit_count for child in children])
        selection = weights - counts / (1 + counts.sum())
        best = int(selection.argmax())
        return actions[best], children[best]

    def _backprop(self, path: list[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = node.reward + self.gamma**node.duration * value


class _GumbelRootSchedule:
    """Allocate a fixed root budget by Gumbel top-k and sequential halving."""

    def __init__(self, root, logits, root_value, rng, cfg):
        self.root = root
        self.logits = logits
        self.root_value = float(root_value)
        self.gamma = cfg.gamma
        self.scale = cfg.gumbel_q_scale
        self.gumbel = rng.gumbel(size=len(logits))
        legal = list(root.children)
        count = min(len(legal), cfg.gumbel_candidates, max(1, cfg.simulations // 2))
        initial = sorted(legal, key=lambda a: self.gumbel[a] + logits[a], reverse=True)
        self.active = initial[:count]
        self.remaining = cfg.simulations
        self.rounds = max(1, math.ceil(math.log2(count)))
        self.start = {action: 0 for action in self.active}
        self.quota = self._quota()

    def _quota(self):
        return max(1, self.remaining // (len(self.active) * self.rounds))

    def completed_q(self):
        return {
            action: (
                child.reward + self.gamma**child.duration * child.value
                if child.visit_count else self.root_value
            )
            for action, child in self.root.children.items()
        }

    def improved_logits(self):
        q = self.completed_q()
        low, high = min(q.values()), max(q.values())
        inv = 1 / (high - low) if high > low + 1e-8 else 0.0
        return {
            action: self.logits[action] + self.scale * (value - low) * inv
            for action, value in q.items()
        }

    def next_action(self):
        while len(self.active) > 1 and all(
            self.root.children[action].visit_count - self.start[action] >= self.quota
            for action in self.active
        ):
            improved = self.improved_logits()
            keep = (len(self.active) + 1) // 2
            self.active = sorted(
                self.active,
                key=lambda action: improved[action] + self.gumbel[action],
                reverse=True,
            )[:keep]
            self.rounds = max(1, self.rounds - 1)
            self.start = {
                action: self.root.children[action].visit_count for action in self.active
            }
            self.quota = self._quota()
        self.remaining -= 1
        if len(self.active) == 1:
            return self.active[0]
        return min(
            self.active,
            key=lambda action: (
                self.root.children[action].visit_count - self.start[action],
                -self.gumbel[action] - self.logits[action],
            ),
        )


@torch.inference_mode()
def search_batch(
    model,
    states,
    remaining,
    cfg,
    rng,
    explore=False,
    *,
    num_bands=MAX_BANDS,
    current_bands=None,
    retune_tables=None,
    return_actions=False,
):
    """Batched search with a physical-tick horizon and macro-action backups."""
    if len(states) != len(remaining) or not len(states) or min(remaining) < 1:
        raise ValueError("one positive remaining horizon per root is required")
    if not 1 <= num_bands <= MAX_BANDS:
        raise ValueError("num_bands must be between one and MAX_BANDS")
    helper = MCTS(
        model, num_bands, num_simulations=cfg.simulations, gamma=cfg.gamma, max_depth=cfg.depth
    )
    logits, root_network_values = model.predict(states)
    action_count = logits.shape[-1]
    dwell_steps = tuple(
        getattr(model, "dwell_steps", (1,) if action_count == MAX_BANDS else DWELL_STEPS)
    )
    if action_count != MAX_BANDS * len(dwell_steps) or (
        cfg.physical_contract and dwell_steps != tuple(cfg.dwell_steps)
    ):
        raise ValueError("search action count or dwell grid differs from model")
    if cfg.physical_contract:
        if current_bands is None or retune_tables is None:
            raise ValueError("physical search requires previous bands and retune timing")
        if len(current_bands) != len(states) or len(retune_tables) != len(states):
            raise ValueError("one retune timing table per search root is required")
        if any(not -1 <= int(band) < num_bands for band in current_bands):
            raise ValueError("previous band is outside the search interface")
        if any(np.asarray(table).shape != (num_bands, num_bands) for table in retune_tables):
            raise ValueError("retune table dimensions differ from search bands")
    else:
        current_bands = [-1] * len(states)
        retune_tables = [None] * len(states)
    duration_count = len(dwell_steps)
    all_actions = tuple(range(num_bands * duration_count))
    base_durations = np.tile(np.asarray(dwell_steps, dtype=np.int64), num_bands)
    destination_bands = np.repeat(np.arange(num_bands), duration_count)
    duration_grids = []
    for table in retune_tables:
        grid = np.broadcast_to(base_durations, (num_bands + 1, len(base_durations))).copy()
        if table is not None:
            grid[1:] += np.asarray(table)[:, destination_bands]
        duration_grids.append(grid)

    def make_children(probabilities, elapsed, ticks_left, previous_band, duration_grid):
        durations = duration_grid[previous_band + 1]
        actions = [
            a for a, duration in zip(all_actions, durations, strict=True) if duration <= ticks_left
        ]
        if not actions:
            return {}
        total = sum(float(probabilities[a]) for a in actions)
        # A masked policy can underflow to zero when all short actions have
        # very low logits; retain a valid normalized prior in that case.
        inv_total = 1.0 / total if total > 0 else 0.0
        uniform = 1.0 / len(actions)
        return {
            a: MCTSNode(
                float(probabilities[a]) * inv_total if total > 0 else uniform,
                duration=durations[a],
                elapsed=elapsed + durations[a],
                band=a // duration_count,
            )
            for a in actions
        }

    def tick_bound(ticks):
        return (1 - cfg.gamma**ticks) / (1 - cfg.gamma)

    priors = logits.softmax(-1).cpu().numpy()
    raw_logits = logits.cpu().numpy()
    root_network_values = root_network_values.cpu().numpy()
    roots = []
    schedules = []
    bounds = [[math.inf, -math.inf] for _ in states]
    horizons = [int(horizon) for horizon in remaining]
    for i, state in enumerate(states):
        root = MCTSNode(1.0, band=int(current_bands[i]))
        root.latent_state = state
        p = priors[i]
        if explore and cfg.search_method == "puct":
            p = 0.75 * p + 0.25 * rng.dirichlet(np.full(action_count, 0.3))
        root.children = make_children(p, 0, horizons[i], root.band, duration_grids[i])
        if cfg.search_method == "gumbel":
            schedules.append(
                _GumbelRootSchedule(root, raw_logits[i], root_network_values[i], rng, cfg)
            )
        roots.append(root)
    for _ in range(cfg.simulations):
        paths, pending, actions, parent_states = [], [], [], []
        for i, root in enumerate(roots):
            if schedules:
                action = schedules[i].next_action()
                node, path = root.children[action], [root, root.children[action]]
            else:
                node, path = root, [root]
            while node.expanded and node.children and len(path) - 1 < cfg.depth:
                if schedules:
                    action, node = helper._select_gumbel_child(
                        node, cfg.gamma, cfg.gumbel_q_scale
                    )
                else:
                    action, node = helper._select_child(
                        node, bounds[i] if cfg.normalize_search else None
                    )
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
            # A terminal leaf has no future value or children. In the common
            # case where every world reaches its terminal step together, this
            # avoids a prediction-network pass entirely.
            active = [j for j, i in enumerate(pending) if horizons[i] > paths[i][-1].elapsed]
            if active:
                child_logits, network_values = model.predict(ns[active])
                probabilities = child_logits.softmax(-1)
                # Concatenate even when only some worlds need a continuation:
                # one transfer/synchronization for the whole inference batch.
                outputs = (
                    torch.cat(
                        (rewards.flatten(), probabilities.flatten(), network_values.flatten())
                    )
                    .cpu()
                    .numpy()
                )
                n = len(pending)
                m = len(active)
                reward_values = outputs[:n]
                predictions = outputs[n : n + m * action_count].reshape(m, action_count)
                values = outputs[n + m * action_count :]
            else:
                reward_values = rewards.cpu().numpy()
                predictions = values = None
            active_rows = {j: row for row, j in enumerate(active)}
            for j, i in enumerate(pending):
                leaf = paths[i][-1]
                leaf.latent_state = ns[j]
                reward_bound = tick_bound(leaf.duration)
                leaf.reward = float(np.clip(reward_values[j], -0.25 * reward_bound, reward_bound))
                row = active_rows.get(j)
                if row is None:
                    leaf.estimate = 0.0
                else:
                    leaf.estimate = float(values[row])
                    if len(paths[i]) - 1 < cfg.depth:
                        leaf.children = make_children(
                            predictions[row],
                            leaf.elapsed,
                            horizons[i] - leaf.elapsed,
                            leaf.band,
                            duration_grids[i],
                        )
        for i, path in enumerate(paths):
            steps_left = horizons[i] - path[-1].elapsed
            if steps_left <= 0:
                value = 0.0
            else:
                value = path[-1].estimate
                if value is None:
                    raise RuntimeError("expanded search leaf has no cached value estimate")
            bound = tick_bound(max(0, steps_left))
            value = float(np.clip(value, -0.25 * bound, bound))
            helper._backprop(path, value)
            if cfg.normalize_search:
                for node in path[1:]:
                    q = node.reward + cfg.gamma**node.duration * node.value
                    bounds[i][0] = min(bounds[i][0], q)
                    bounds[i][1] = max(bounds[i][1], q)
    visits = np.zeros((len(roots), action_count), dtype=np.float32)
    selected = np.empty(len(roots), dtype=np.int64)
    for i, root in enumerate(roots):
        if schedules:
            improved = schedules[i].improved_logits()
            actions = list(improved)
            scores = np.asarray([improved[action] for action in actions])
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
            visits[i, actions] = scores
            selected[i] = max(
                schedules[i].active,
                key=lambda action: improved[action] + schedules[i].gumbel[action],
            )
        else:
            for action, child in root.children.items():
                visits[i, action] = child.visit_count
            visits[i] /= visits[i].sum()
            selected[i] = int(visits[i].argmax())
    # Root value is the mean backed-up return, not a privileged simulator target.
    values = np.array([r.value for r in roots], dtype=np.float32)
    return (visits, values, selected) if return_actions else (visits, values)
