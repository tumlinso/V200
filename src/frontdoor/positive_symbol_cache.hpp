#pragma once

#include <atomic>

namespace v200::frontdoor::detail {

// A call-site tag gives each cached symbol its own process-lifetime slot.
// Only successful lookups are cached so lazy CUDA/V200 loading keeps working.
template <class Tag, class Function>
std::atomic<Function> &positive_symbol_slot() noexcept {
  static std::atomic<Function> cached{nullptr};
  return cached;
}

template <class Tag, class Function, class Resolver>
Function positive_symbol(Resolver &&resolve) noexcept {
  auto &cached = positive_symbol_slot<Tag, Function>();
  Function function = cached.load(std::memory_order_acquire);
  if (function != nullptr) return function;

  function = resolve();
  if (function == nullptr) return nullptr;

  Function expected = nullptr;
  if (cached.compare_exchange_strong(expected, function,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
    return function;
  }
  return expected;
}

}  // namespace v200::frontdoor::detail
