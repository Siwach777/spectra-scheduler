import pytest

from spectra_scheduler.emitters import SpatialScanningEmitter
from spectra_scheduler.scenario_io import build_scenario_from_definition
from spectra_scheduler.scenarios import REQUIREMENT_SCENARIOS, build_scenario


def test_spatial_scan_fixed_carrier_and_independent_pulse_clock():
    emitter = SpatialScanningEmitter("beam", 2, 10, 4, pulse_period=3, phase=8)
    events = emitter.transmissions(31, 4)
    # Beam windows [0,2), [8,12), [18,22), [28,31); pulse times multiples of 3.
    assert [e.time_step for e in events] == [0, 9, 18, 21, 30]
    assert {e.band for e in events} == {2}


@pytest.mark.parametrize(
    "changes",
    [
        dict(scan_period=0),
        dict(visible_steps=11),
        dict(visible_steps=0),
        dict(pulse_period=True),
        dict(phase=-1),
        dict(band=4),
    ],
)
def test_spatial_invalid_parameters(changes):
    with pytest.raises(ValueError):
        SpatialScanningEmitter(
            **(dict(emitter_id="beam", band=2, scan_period=10, visible_steps=2) | changes)
        ).transmissions(20, 4)


def test_all_requirement_scenarios_reproducible_and_distinct():
    for name in REQUIREMENT_SCENARIOS:
        a, b = build_scenario(name, 11), build_scenario(name, 11)
        assert a.generate_truth() == b.generate_truth()
        assert a.generate_truth()
        assert a.generate_truth() != build_scenario(name, 12).generate_truth()
        if name == "frequency-agile":
            assert len({e.band for e in a.generate_truth()}) == 8
        else:
            for emitter in a.emitters:
                assert len({e.band for e in emitter.transmissions(a.duration, a.num_bands)}) == 1


def test_focused_spatial_carriers_and_periodic_clocks_vary_across_worlds():
    spatial = [build_scenario("spatial-scan", seed) for seed in range(20)]
    periodic = [build_scenario("periodic-scan", seed) for seed in range(20)]
    assert len({tuple(e.band for e in world.emitters) for world in spatial}) > 1
    assert len({world.emitters[0].band for world in periodic}) > 1
    assert len({world.emitters[0].scan_period for world in periodic}) > 1
    assert len({world.emitters[0].pulse_period for world in periodic}) > 1


def test_custom_spatial_scan_random_phase():
    spec = {
        "num_bands": 4,
        "duration": 30,
        "emitters": [
            {
                "type": "spatial-scanning",
                "emitter_id": "beam",
                "band": 1,
                "scan_period": 10,
                "visible_steps": 2,
                "phase": "random",
            }
        ],
    }
    assert build_scenario_from_definition(spec, 1).generate_truth()


def test_periodic_aliasing_case_has_blind_phases():
    from spectra_scheduler.receiver import Receiver
    from spectra_scheduler.schedulers import RoundRobinScheduler
    from spectra_scheduler.simulation import Simulation

    captures = []
    for phase in range(16):
        world = Simulation(
            8, 128, (SpatialScanningEmitter("beam", 3, 16, 2, phase=phase),), Receiver()
        )
        result = world.run(RoundRobinScheduler())
        captures.append(sum(len(r.detected_emitters) for r in result.detection_records))
    assert min(captures) == 0 and max(captures) > 0


def test_physical_periodic_visibility_varies_hidden_clocks_and_is_split_isolated():
    from spectra_scheduler.rl_scenarios import physical_periodic_visibility_scenario

    worlds = [physical_periodic_visibility_scenario(seed, "validation") for seed in range(20)]
    emitters = [world.emitters[0] for world in worlds]
    assert all(isinstance(emitter, SpatialScanningEmitter) for emitter in emitters)
    assert len({emitter.scan_period for emitter in emitters}) > 1
    assert len({emitter.visible_steps for emitter in emitters}) > 1
    assert len({emitter.pulse_period for emitter in emitters}) > 1
    test_world = physical_periodic_visibility_scenario(0, "test")
    assert worlds[0].generate_truth() != test_world.generate_truth()


def test_physical_family_labels_frequency_sweep_correctly():
    from spectra_scheduler.emitters import ScanningEmitter
    from spectra_scheduler.receiver import Receiver
    from spectra_scheduler.rl_benchmark import emitter_family_counts
    from spectra_scheduler.schedulers import RoundRobinScheduler
    from spectra_scheduler.simulation import Simulation

    world = Simulation(2, 6, (ScanningEmitter("sweep", 0, 1, 1),), Receiver())
    families = emitter_family_counts(world, world.run(RoundRobinScheduler()))
    assert "frequency-scan" in families
    assert "periodic-scan" not in families

    periodic = Simulation(2, 6, (SpatialScanningEmitter("beam", 1, 3, 2),), Receiver())
    periodic_result = periodic.run(RoundRobinScheduler())
    families = emitter_family_counts(periodic, periodic_result, focused_periodic=True)
    assert "periodic-scan" in families
