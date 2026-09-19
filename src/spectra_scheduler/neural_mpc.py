"""Compatibility entry point; implementation lives in spectra_scheduler.mpc."""

from spectra_scheduler.mpc.checkpoints import load_model as load_model
from spectra_scheduler.mpc.checkpoints import save_model as save_model
from spectra_scheduler.mpc.cli import demonstration_main as main
from spectra_scheduler.mpc.config import DEFAULT_COVERAGE_LIMIT as DEFAULT_COVERAGE_LIMIT
from spectra_scheduler.mpc.config import DEFAULT_GAMMA as DEFAULT_GAMMA
from spectra_scheduler.mpc.config import DEFAULT_MCTS_SIMS as DEFAULT_MCTS_SIMS
from spectra_scheduler.mpc.config import GRU_HIDDEN as GRU_HIDDEN
from spectra_scheduler.mpc.config import MAX_BANDS as MAX_BANDS
from spectra_scheduler.mpc.config import MODEL_VERSION as MODEL_VERSION
from spectra_scheduler.mpc.config import STEP_FEATURE_DIM as STEP_FEATURE_DIM
from spectra_scheduler.mpc.config import TrainConfig as TrainConfig
from spectra_scheduler.mpc.data import _run_and_record as _run_and_record
from spectra_scheduler.mpc.data import collect_demonstrations as collect_demonstrations
from spectra_scheduler.mpc.evaluation import _benchmark as _benchmark
from spectra_scheduler.mpc.learning import _train_on_episode as _train_on_episode
from spectra_scheduler.mpc.learning import train_model as train_model
from spectra_scheduler.mpc.model import DynamicsNetwork as DynamicsNetwork
from spectra_scheduler.mpc.model import NeuralMPCModel as NeuralMPCModel
from spectra_scheduler.mpc.model import PredictionNetwork as PredictionNetwork
from spectra_scheduler.mpc.model import RepresentationNetwork as RepresentationNetwork
from spectra_scheduler.mpc.observation import ObservationEncoder as ObservationEncoder
from spectra_scheduler.mpc.observation import RewardFunction as RewardFunction
from spectra_scheduler.mpc.scheduler import NeuralMPCScheduler as NeuralMPCScheduler
from spectra_scheduler.mpc.search import MCTS as MCTS
from spectra_scheduler.mpc.search import MCTSNode as MCTSNode

if __name__ == "__main__":
    main()
