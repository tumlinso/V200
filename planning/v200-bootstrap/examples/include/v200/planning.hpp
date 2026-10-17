#pragma once
// CPU-only reference logic. This is not a CUDA allocator or a kernel rewriter.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace v200 {
using u64 = std::uint64_t;
inline u64 add(u64 a, u64 b) {
    if (b > std::numeric_limits<u64>::max() - a) throw std::overflow_error("addition");
    return a + b;
}
inline u64 mul(u64 a, u64 b) {
    if (a && b > std::numeric_limits<u64>::max() / a) throw std::overflow_error("multiplication");
    return a * b;
}
inline u64 round_up(u64 n, u64 g) {
    if (!g) throw std::invalid_argument("zero granularity");
    return n % g ? add(n, g - n % g) : n;
}
inline u64 common_granularity(u64 a, u64 b) {
    if (!a || !b) throw std::invalid_argument("zero granularity");
    return mul(a / std::gcd(a, b), b);
}
struct Extent { u64 offset, bytes; unsigned owner; };
struct PairPlan { u64 payload, mapped, granularity; std::vector<Extent> extents; };

// Balancing is a baseline, not the recommended policy for every application.
// budgets are already net of reserves. Each nonempty extent is whole-granule.
inline PairPlan plan_pair(u64 payload, u64 gran_a, u64 gran_b, u64 budget_a, u64 budget_b) {
    if (!payload) throw std::invalid_argument("zero payload");
    const auto g = common_granularity(gran_a, gran_b);
    const auto mapped = round_up(payload, g);
    const auto units = mapped / g, cap_a = budget_a / g, cap_b = budget_b / g;
    if (units > cap_a && units - cap_a > cap_b) throw std::length_error("insufficient pair capacity");
    auto ua = std::min(units / 2 + units % 2, cap_a);
    if (units - ua > cap_b) ua = units - cap_b;
    const auto ub = units - ua;
    PairPlan p{payload, mapped, g, {}};
    const auto ba = mul(ua, g), bb = mul(ub, g);
    if (ba) p.extents.push_back({0, ba, 0});
    if (bb) p.extents.push_back({ba, bb, 1});
    return p;
}

// Split an application byte interval into canonical owner intervals. Does not
// claim a CUDA copy or mapping at arbitrary offsets is permitted.
inline std::vector<Extent> intersections(const PairPlan& p, u64 offset, u64 bytes) {
    const auto end = add(offset, bytes);
    if (end > p.payload) throw std::out_of_range("interval outside payload");
    std::vector<Extent> out;
    for (const auto& e : p.extents) {
        const auto lo = std::max(offset, e.offset), hi = std::min(end, add(e.offset, e.bytes));
        if (lo < hi) out.push_back({lo, hi - lo, e.owner});
    }
    return out;
}
struct GridPart { u64 first_block, block_count; };
inline std::vector<GridPart> split_grid(u64 blocks) {
    const auto a = blocks / 2 + blocks % 2;
    std::vector<GridPart> out;
    if (a) out.push_back({0, a});
    if (blocks > a) out.push_back({a, blocks - a});
    return out;
}
// Reference enumeration of a split grid-stride loop. Real GPU code must retain
// the logical stride and have a separate split-safety/aliasing contract.
inline std::vector<u64> visited(u64 items, u64 logical_blocks, u64 threads, GridPart p) {
    if (!threads || !logical_blocks || add(p.first_block, p.block_count) > logical_blocks)
        throw std::invalid_argument("invalid grid");
    const auto stride = mul(logical_blocks, threads);
    std::vector<u64> out;
    for (u64 b = 0; b < p.block_count; ++b) {
        const auto base = mul(add(p.first_block, b), threads);
        for (u64 t = 0; t < threads; ++t) {
            auto i = add(base, t);
            while (i < items) {
                out.push_back(i);
                if (items - i <= stride) break; // also avoids overflow
                i = add(i, stride);
            }
        }
    }
    return out;
}
} // namespace v200
