#include "vmm_location.hpp"

#include <cstdlib>
#include <iostream>

using v200::frontdoor::detail::NativePairLocation;
using v200::frontdoor::detail::map_external_vmm_device;

namespace {
int failures = 0;

void expect_mapping(int requested, const int *devices, int count,
                    int expected_pair, int expected_device,
                    const char *message) {
    NativePairLocation mapped;
    const bool ok = map_external_vmm_device(requested, devices, count, &mapped);
    if (!ok || mapped.pair != expected_pair || mapped.device != expected_device) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_rejected(int requested, const int *devices, int count,
                     const char *message) {
    NativePairLocation mapped;
    if (map_external_vmm_device(requested, devices, count, &mapped)) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
}

int main() {
    // Simulates the native ordinals produced by CUDA_VISIBLE_DEVICES ordered
    // as pair0 anchor/peer, pair1 anchor/peer.
    const int four_devices[] = {0, 1, 2, 3};
    expect_mapping(0, four_devices, 4, 0, 0,
                   "logical ordinal zero maps to pair-zero anchor");
    expect_mapping(1, four_devices, 4, 1, 2,
                   "logical ordinal one takes precedence over hidden peer one");
    expect_mapping(2, four_devices, 4, 1, 2,
                   "published native pair-one anchor remains accepted");
    expect_rejected(3, four_devices, 4,
                    "hidden pair-one peer is not an external VMM device");
    expect_rejected(-1, four_devices, 4, "negative device is rejected");

    // Native CUdevice handles need not be equal to their visible ordinals.
    const int reordered_devices[] = {10, 11, 20, 21};
    expect_mapping(0, reordered_devices, 4, 0, 10,
                   "logical ordinal maps through UUID-selected anchor");
    expect_mapping(1, reordered_devices, 4, 1, 20,
                   "second logical ordinal maps through selected anchor");
    expect_mapping(20, reordered_devices, 4, 1, 20,
                   "native handle returned for pair-one anchor is accepted");
    expect_rejected(11, reordered_devices, 4,
                    "hidden native peer handle is rejected");
    expect_rejected(21, reordered_devices, 4,
                    "second hidden native peer handle is rejected");

    const int single_pair[] = {7, 8};
    expect_mapping(0, single_pair, 2, 0, 7,
                   "single-pair logical ordinal maps to its anchor");
    expect_mapping(7, single_pair, 2, 0, 7,
                   "single-pair native anchor remains accepted");
    expect_rejected(8, single_pair, 2,
                    "single-pair hidden peer is not externally published");
    expect_rejected(0, four_devices, 3, "unsupported selection count is rejected");
    expect_rejected(0, nullptr, 4, "missing selected-device list is rejected");
    if (map_external_vmm_device(0, four_devices, 4, nullptr)) {
        std::cerr << "FAIL: null result pointer is rejected\n";
        ++failures;
    }

    if (failures == 0) std::cout << "VMM device mapping checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
