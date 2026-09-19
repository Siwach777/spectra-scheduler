import unittest

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.models import Transmission


class PeriodicEmitterTests(unittest.TestCase):
    def test_generates_events_until_duration(self) -> None:
        emitter = PeriodicEmitter("search-emitter", band=2, period=3, phase=1)

        events = emitter.transmissions(duration=11, num_bands=4)

        self.assertEqual(
            events,
            [
                Transmission(1, 2, "search-emitter"),
                Transmission(4, 2, "search-emitter"),
                Transmission(7, 2, "search-emitter"),
                Transmission(10, 2, "search-emitter"),
            ],
        )

    def test_rejects_band_outside_spectrum(self) -> None:
        emitter = PeriodicEmitter("invalid", band=4, period=2)

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=5, num_bands=4)

    def test_attaches_signal_properties_to_events(self) -> None:
        emitter = PeriodicEmitter(
            "weak-emitter",
            band=1,
            period=2,
            power_dbm=-87.5,
            pulse_width_us=1.8,
        )

        events = emitter.transmissions(duration=3, num_bands=2)

        self.assertTrue(all(event.power_dbm == -87.5 for event in events))
        self.assertTrue(all(event.pulse_width_us == 1.8 for event in events))

    def test_rejects_invalid_pulse_width(self) -> None:
        emitter = PeriodicEmitter(
            "invalid",
            band=1,
            period=2,
            pulse_width_us=0.0,
        )

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=3, num_bands=2)


class FrequencyHoppingEmitterTests(unittest.TestCase):
    def test_repeats_hop_sequence(self) -> None:
        emitter = FrequencyHoppingEmitter("hopper", bands=(0, 2, 1), period=2)

        events = emitter.transmissions(duration=9, num_bands=3)

        self.assertEqual([event.time_step for event in events], [0, 2, 4, 6, 8])
        self.assertEqual([event.band for event in events], [0, 2, 1, 0, 2])

    def test_requires_at_least_one_band(self) -> None:
        emitter = FrequencyHoppingEmitter("hopper", bands=(), period=2)

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=5, num_bands=3)


class ScanningEmitterTests(unittest.TestCase):
    def test_scans_adjacent_bands_and_reverses_at_edges(self) -> None:
        emitter = ScanningEmitter(
            "scanner",
            lowest_band=1,
            highest_band=3,
            period=1,
        )

        events = emitter.transmissions(duration=9, num_bands=5)

        self.assertEqual([event.band for event in events], [1, 2, 3, 2, 1, 2, 3, 2, 1])

    def test_rejects_reversed_scan_range(self) -> None:
        emitter = ScanningEmitter(
            "scanner",
            lowest_band=3,
            highest_band=1,
            period=1,
        )

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=5, num_bands=5)


class BurstEmitterTests(unittest.TestCase):
    def test_generates_pulses_in_separate_bursts(self) -> None:
        emitter = BurstEmitter(
            "burst",
            band=1,
            burst_period=6,
            pulses_per_burst=3,
            pulse_spacing=1,
            phase=1,
        )

        events = emitter.transmissions(duration=10, num_bands=3)

        self.assertEqual([event.time_step for event in events], [1, 2, 3, 7, 8, 9])

    def test_rejects_overlapping_bursts(self) -> None:
        emitter = BurstEmitter("burst", band=1, burst_period=4, pulses_per_burst=3, pulse_spacing=2)

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=10, num_bands=3)


class JitteredPeriodicEmitterTests(unittest.TestCase):
    def test_uses_repeatable_bounded_intervals(self) -> None:
        emitter = JitteredPeriodicEmitter("jittered", band=2, period=5, jitter=2, seed=9)

        first = emitter.transmissions(duration=30, num_bands=3)
        second = emitter.transmissions(duration=30, num_bands=3)
        intervals = [
            later.time_step - earlier.time_step for earlier, later in zip(first, first[1:])
        ]

        self.assertEqual(first, second)
        self.assertTrue(all(3 <= interval <= 7 for interval in intervals))

    def test_requires_jitter_smaller_than_period(self) -> None:
        emitter = JitteredPeriodicEmitter("jittered", band=2, period=4, jitter=4)

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=10, num_bands=3)


class WindowedEmitterTests(unittest.TestCase):
    def test_limits_emitter_to_active_window(self) -> None:
        emitter = WindowedEmitter(
            PeriodicEmitter("late", band=1, period=2),
            start_time=3,
            end_time=8,
        )

        events = emitter.transmissions(duration=10, num_bands=3)

        self.assertEqual([event.time_step for event in events], [4, 6])

    def test_rejects_empty_window(self) -> None:
        emitter = WindowedEmitter(
            PeriodicEmitter("invalid", band=1, period=2),
            start_time=5,
            end_time=5,
        )

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=10, num_bands=3)


class ModeSwitchingEmitterTests(unittest.TestCase):
    def test_changes_pattern_at_switch_time(self) -> None:
        emitter = ModeSwitchingEmitter(
            first_mode=PeriodicEmitter("changing", band=0, period=2),
            second_mode=PeriodicEmitter("changing", band=2, period=3, phase=1),
            switch_time=4,
        )

        events = emitter.transmissions(duration=9, num_bands=3)

        self.assertEqual(
            events,
            [
                Transmission(0, 0, "changing"),
                Transmission(2, 0, "changing"),
                Transmission(4, 2, "changing"),
                Transmission(7, 2, "changing"),
            ],
        )

    def test_requires_both_modes_to_use_same_identity(self) -> None:
        emitter = ModeSwitchingEmitter(
            first_mode=PeriodicEmitter("first", band=0, period=2),
            second_mode=PeriodicEmitter("second", band=1, period=2),
            switch_time=3,
        )

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=8, num_bands=3)


if __name__ == "__main__":
    unittest.main()
