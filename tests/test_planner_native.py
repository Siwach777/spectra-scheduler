import shutil

import numpy as np
import pytest

from spectra_scheduler.planner_native import build
from spectra_scheduler.timing_planner import ForecastPlannerWorkspace, first_actions


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if not shutil.which("c++"):
        pytest.skip("optional native compiler unavailable")
    return build(tmp_path_factory.mktemp("planner") / "planner.so")


def test_native_values_match_numpy_variable_horizons_and_ties(library, monkeypatch):
    rng = np.random.default_rng(11)
    for bands in (2, 8):
        for zero in (False, True):
            predicted = rng.random((4, bands, 80))
            if zero:
                predicted.fill(0)
            retune = rng.integers(1, 7, size=(4, bands, bands))
            retune[:, np.arange(bands), np.arange(bands)] = 0
            current = np.array([-1, 0, bands - 1, 1])
            horizons = np.array([1, 17, 79, 200])
            original = ForecastPlannerWorkspace(4, bands, 80)
            native = ForecastPlannerWorkspace(4, bands, 80)
            monkeypatch.delenv("SPECTRA_PLANNER_LIBRARY", raising=False)
            expected = first_actions(predicted, current, retune, horizons, workspace=original)
            monkeypatch.setenv("SPECTRA_PLANNER_LIBRARY", str(library))
            actual = first_actions(predicted, current, retune, horizons, workspace=native)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(native.action_values, original.action_values, atol=1e-12)
            np.testing.assert_allclose(native.state_values, original.state_values, atol=1e-12)
