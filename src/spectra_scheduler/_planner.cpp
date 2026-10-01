#include <algorithm>
#include <cstdint>
#include <limits>

extern "C" std::int64_t spectra_planner_version() { return 1; }

extern "C" void spectra_plan(
    std::int64_t batch, std::int64_t bands, std::int64_t future, std::int64_t dwells,
    const double* prefix, const std::int64_t* delays, const std::int64_t* menu,
    const std::int64_t* horizons, double* values, double* first_q) {
    const auto states = bands + 1;
    for (std::int64_t b = 0; b < batch; ++b) {
        const auto limit = horizons[b];
        auto state = values + b * (future + 1) * states;
        auto q = first_q + b * states * bands * dwells;
        for (std::int64_t t = future - 1; t >= 0; --t) {
            for (std::int64_t previous = 0; previous < states; ++previous) {
                double best = -std::numeric_limits<double>::infinity();
                for (std::int64_t band = 0; band < bands; ++band) {
                    const auto delay = delays[(b * states + previous) * bands + band];
                    const auto start = std::min(limit, t + delay);
                    const auto counts = prefix + (b * bands + band) * (future + 1);
                    for (std::int64_t d = 0; d < dwells; ++d) {
                        const auto end = std::min(limit, start + menu[d]);
                        const double score = counts[end] - counts[start] + state[end * states + band];
                        best = std::max(best, score);
                        if (t == 0) q[(previous * bands + band) * dwells + d] = score;
                    }
                }
                state[t * states + previous] = best;
            }
        }
    }
}
