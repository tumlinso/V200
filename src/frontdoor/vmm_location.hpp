#pragma once

namespace v200::frontdoor::detail {

struct NativePairLocation {
    int pair = -1;
    int device = -1;
};

// V200 presents one logical device for each consecutive pair of selected
// native devices. External ordinals therefore take precedence over native
// CUdevice values (logical 1 can numerically equal pair 0's hidden peer).
// Outside the logical ordinal range, only published pair anchors are valid.
inline bool map_external_vmm_device(int requested_device,
                                    const int *selected_devices,
                                    int selected_count,
                                    NativePairLocation *mapped) noexcept {
    if (mapped == nullptr || selected_devices == nullptr ||
        (selected_count != 2 && selected_count != 4)) return false;

    const int pair_count = selected_count / 2;
    if (requested_device >= 0 && requested_device < pair_count) {
        mapped->pair = requested_device;
        mapped->device = selected_devices[requested_device * 2];
        return true;
    }

    for (int pair = 0; pair < pair_count; ++pair) {
        const int anchor = selected_devices[pair * 2];
        if (requested_device == anchor) {
            mapped->pair = pair;
            mapped->device = anchor;
            return true;
        }
    }
    return false;
}

} // namespace v200::frontdoor::detail
