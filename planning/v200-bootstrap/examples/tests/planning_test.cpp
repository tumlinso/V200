#include "v200/planning.hpp"
#include <functional>
#include <iostream>
#include <string>
using namespace v200;
static unsigned checks = 0;
static void require(bool ok, const char* what) { ++checks; if (!ok) throw std::runtime_error(what); }
template<class E> static void rejects(const std::function<void()>& fn) {
    bool rejected = false; try { fn(); } catch (const E&) { rejected = true; }
    require(rejected, "expected exception");
}
int main() {
    try {
        require(common_granularity(6, 8) == 24, "LCM");
        require(round_up(24, 8) == 24 && round_up(25, 8) == 32, "rounding");
        rejects<std::invalid_argument>([] { round_up(2, 0); });
        rejects<std::overflow_error>([] { add(UINT64_MAX, 1); });
        rejects<std::overflow_error>([] { mul(UINT64_MAX, 2); });
        rejects<std::overflow_error>([] { round_up(UINT64_MAX, 8); });
        rejects<std::overflow_error>([] { common_granularity(UINT64_MAX, 2); });
        rejects<std::invalid_argument>([] { plan_pair(0, 8, 8, 64, 64); });
        auto p = plan_pair(99, 8, 16, 64, 64);
        require(p.mapped == 112 && p.granularity == 16, "mapped extent");
        require(p.extents.size() == 2 && p.extents[0].bytes == 64 && p.extents[1].bytes == 48, "balanced owners");
        auto s = intersections(p, 60, 12);
        require(s.size() == 2 && s[0].bytes == 4 && s[1].bytes == 8, "boundary slicing");
        require(intersections(p, 99, 0).empty(), "empty valid interval");
        rejects<std::out_of_range>([&] { intersections(p, 99, 1); });
        rejects<std::overflow_error>([&] { intersections(p, UINT64_MAX, 1); });
        rejects<std::length_error>([] { plan_pair(129, 16, 16, 64, 64); });
        p = plan_pair(80, 8, 8, 16, 96);
        require(p.extents[0].bytes == 16 && p.extents[1].bytes == 64, "unequal budget");
        p = plan_pair(5, 8, 8, 0, 8);
        require(p.extents.size() == 1 && p.extents[0].owner == 1 && p.extents[0].offset == 0, "one owner");
        require(split_grid(0).empty(), "zero grid");
        for (u64 blocks : {1ULL, 2ULL, 3ULL, 7ULL}) {
            for (u64 items : {0ULL, 1ULL, 31ULL, 32ULL, 33ULL, 257ULL}) {
                std::vector<u64> all;
                for (auto part : split_grid(blocks)) {
                    auto v = visited(items, blocks, 32, part);
                    all.insert(all.end(), v.begin(), v.end());
                }
                std::sort(all.begin(), all.end());
                require(all.size() == items, "coverage size");
                for (u64 i = 0; i < items; ++i) require(all[i] == i, "exactly once logical coverage");
            }
        }
        rejects<std::invalid_argument>([] { visited(8, 2, 32, {2, 1}); });
        rejects<std::invalid_argument>([] { visited(8, 2, 0, {0, 1}); });
        rejects<std::overflow_error>([] { visited(8, UINT64_MAX, 2, {0, 1}); });
        std::cout << "{\"status\":\"CPU_PASS\",\"checks\":" << checks << ",\"gpu_evidence\":false}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
