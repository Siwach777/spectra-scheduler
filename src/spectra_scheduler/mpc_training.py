"""Compatibility entry point; implementation lives in spectra_scheduler.mpc."""

from spectra_scheduler.mpc.checkpoints import atomic_json as atomic_json
from spectra_scheduler.mpc.checkpoints import save_training as save_training
from spectra_scheduler.mpc.cli import training_main as main
from spectra_scheduler.mpc.config import REWARD as REWARD
from spectra_scheduler.mpc.config import VERSION as VERSION
from spectra_scheduler.mpc.config import Config as Config
from spectra_scheduler.mpc.data import Replay as Replay
from spectra_scheduler.mpc.data import collect as collect
from spectra_scheduler.mpc.data import collect_worker as collect_worker
from spectra_scheduler.mpc.evaluation import evaluate_run as evaluate_run
from spectra_scheduler.mpc.evaluation import probe_reward_error as probe_reward_error
from spectra_scheduler.mpc.evaluation import validate as validate
from spectra_scheduler.mpc.learning import batch_loss as batch_loss
from spectra_scheduler.mpc.learning import value_targets as value_targets
from spectra_scheduler.mpc.search import search_batch as search_batch
from spectra_scheduler.mpc.trainer import _run_locked as _run_locked
from spectra_scheduler.mpc.trainer import run as run

if __name__ == "__main__":
    main()
