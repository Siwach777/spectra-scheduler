import unittest

from spectra_scheduler.emitters import FrequencyHoppingEmitter, PeriodicEmitter
from spectra_scheduler.models import Transmission


class PeriodicEmitterTests(unittest.TestCase):
    def test_generates_events_until_duration(self) -> None:
        emitter = PeriodicEmitter("search-radar", band=2, period=3, phase=1)

        events = emitter.transmissions(duration=11, num_bands=4)

        self.assertEqual(
            events,
            [
                Transmission(1, 2, "search-radar"),
                Transmission(4, 2, "search-radar"),
                Transmission(7, 2, "search-radar"),
                Transmission(10, 2, "search-radar"),
            ],
        )

    def test_rejects_band_outside_spectrum(self) -> None:
        emitter = PeriodicEmitter("invalid", band=4, period=2)

        with self.assertRaises(ValueError):
            emitter.transmissions(duration=5, num_bands=4)


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


if __name__ == "__main__":
    unittest.main()
