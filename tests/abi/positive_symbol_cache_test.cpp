#include "positive_symbol_cache.hpp"

#include <cstdlib>
#include <iostream>

namespace {
using Function = int (*)(int);
struct RetryTag {};
struct StableTag {};

int plus_seven(int value) { return value + 7; }
int times_three(int value) { return value * 3; }

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}
}  // namespace

int main() {
  int attempts = 0;
  Function available = nullptr;
  auto resolve_retry = [&]() noexcept -> Function {
    ++attempts;
    return available;
  };

  require(v200::frontdoor::detail::positive_symbol<RetryTag, Function>(resolve_retry) == nullptr,
          "a missing symbol remains a miss");
  require(v200::frontdoor::detail::positive_symbol<RetryTag, Function>(resolve_retry) == nullptr,
          "a cached miss is retried");
  require(attempts == 2, "each positive-cache miss invokes its resolver");

  available = &plus_seven;
  const Function first =
      v200::frontdoor::detail::positive_symbol<RetryTag, Function>(resolve_retry);
  require(first == &plus_seven && first(5) == 12, "first positive lookup is usable");
  int alternate_attempts = 0;
  const Function second =
      v200::frontdoor::detail::positive_symbol<RetryTag, Function>([&]() noexcept {
        ++alternate_attempts;
        return &times_three;
      });
  require(second == first && second(5) == 12,
          "a successful process-lifetime lookup remains stable");
  require(attempts == 3, "a successful lookup avoids another resolver call");
  require(alternate_attempts == 0, "tagged slot is shared across resolver closure types");

  int independent_attempts = 0;
  const Function independent =
      v200::frontdoor::detail::positive_symbol<StableTag, Function>([&]() noexcept {
        ++independent_attempts;
        return &times_three;
      });
  require(independent == &times_three && independent(5) == 15,
          "distinct call-site tags use independent slots");
  require(independent_attempts == 1, "independent slot resolves once");
}
