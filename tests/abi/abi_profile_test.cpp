#include "v200/frontdoor/abi_profile.hpp"

#include <cstdlib>
#include <iostream>

using v200::frontdoor::ApiDisposition;
using v200::frontdoor::classify_api;
using v200::frontdoor::find_api;
using v200::frontdoor::version_available;

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char *message) {
        if (!ok) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    };

    check(classify_api("cuMemMap") == ApiDisposition::wrapped,
          "the consumer's direct Driver VMM mapping call must be wrapped");
    check(classify_api("cudaMallocManaged") == ApiDisposition::unsupported,
          "managed allocations must remain explicitly unsupported in pair mode");
    check(classify_api("cublasGemmEx") == ApiDisposition::unknown,
          "unqualified library calls must not be mislabeled transparent");
    const auto *proc = find_api("cuLaunchKernel");
    check(proc != nullptr && proc->has_per_thread_variant,
          "driver launch lookup must expose a distinct PTDS ABI variant");
    check(proc != nullptr && !version_available(*proc, 3000) && version_available(*proc, 12090),
          "driver entry-point version selection must honor its minimum version");
    check(classify_api("cudaGraphLaunch") == ApiDisposition::native_anchor_passthrough,
          "native graph execution must be explicitly classified as anchor dispatch");
    check(classify_api("cudaMemset") == ApiDisposition::wrapped,
          "synchronous memset must be intercepted for managed-memory lifetime tracking");
    check(classify_api("cuMemGetInfo_v2") == ApiDisposition::wrapped &&
              classify_api("cuDeviceTotalMem_v2") == ApiDisposition::wrapped,
          "Driver memory queries must report the same pair capacity contract");
    check(classify_api("cuMemsetD8Async") == ApiDisposition::wrapped &&
              classify_api("cuMemsetD16Async") == ApiDisposition::wrapped &&
              classify_api("cuMemsetD32Async") == ApiDisposition::wrapped &&
              classify_api("cuMemsetD2D8Async") == ApiDisposition::wrapped &&
              classify_api("cuMemsetD2D16Async") == ApiDisposition::wrapped &&
              classify_api("cuMemsetD2D32Async") == ApiDisposition::wrapped,
          "asynchronous Driver memset variants must retain pair-backed memory through completion");
    check(classify_api("cudaLaunchKernel") == ApiDisposition::wrapped &&
              classify_api("cudaLaunchKernelExC") == ApiDisposition::wrapped,
          "opaque-argument runtime launch entry points must be intercepted for prefix retention");
    check(classify_api("cuGetProcAddress_v2") == ApiDisposition::wrapped,
          "versioned Driver resolver must return frontdoor wrappers for qualified APIs");

    if (failures == 0) std::cout << "ABI profile checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
