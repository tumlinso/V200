#include "v200/memory/manager.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace v200::memory {
namespace {

thread_local uint32_t selected_pair_index = 0;

bool checked_add_u64(CUdeviceptr base, size_t bytes, CUdeviceptr *end) noexcept {
    const auto amount = static_cast<CUdeviceptr>(bytes);
    if (end == nullptr || amount > std::numeric_limits<CUdeviceptr>::max() - base) {
        return false;
    }
    *end = base + amount;
    return true;
}

bool checked_round_up(size_t value, size_t granularity, size_t *out) noexcept {
    if (granularity == 0 || out == nullptr) {
        return false;
    }
    const size_t remainder = value % granularity;
    if (remainder == 0) {
        *out = value;
        return true;
    }
    const size_t extra = granularity - remainder;
    if (extra > std::numeric_limits<size_t>::max() - value) {
        return false;
    }
    *out = value + extra;
    return true;
}

size_t gcd_size(size_t a, size_t b) noexcept {
    while (b != 0) {
        const size_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

bool checked_lcm(size_t a, size_t b, size_t *out) noexcept {
    const size_t gcd = gcd_size(a, b);
    if (gcd == 0 || (a / gcd != 0 && b > std::numeric_limits<size_t>::max() / (a / gcd))) {
        return false;
    }
    *out = (a / gcd) * b;
    return true;
}

bool is_power_of_two(size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool c_string_present(const char *value, size_t capacity) noexcept {
    if (value == nullptr || capacity == 0) {
        return false;
    }
    const void *end = std::memchr(value, '\0', capacity);
    return end != nullptr && value[0] != '\0';
}

int owner_index(const v200_memory_pair_profile &profile, CUdevice device) noexcept {
    if (profile.owner_device[0] == device) {
        return 0;
    }
    if (profile.owner_device[1] == device) {
        return 1;
    }
    return -1;
}

bool driver_complete(const v200_memory_driver_api &driver) noexcept {
    return driver.abi_version == V200_MEMORY_ABI_VERSION &&
           driver.struct_size >= sizeof(v200_memory_driver_api) &&
           driver.mem_address_reserve != nullptr &&
           driver.mem_address_free != nullptr && driver.mem_create != nullptr &&
           driver.mem_release != nullptr && driver.mem_map != nullptr &&
           driver.mem_unmap != nullptr && driver.mem_set_access != nullptr &&
           driver.mem_get_granularity != nullptr;
}

bool profile_valid(const v200_memory_pair_profile &profile) noexcept {
    if (profile.abi_version != V200_MEMORY_ABI_VERSION ||
        profile.struct_size < sizeof(v200_memory_pair_profile) ||
        profile.owner_device[0] == profile.owner_device[1] ||
        profile.owner_granularity[0] == 0 || profile.owner_granularity[1] == 0 ||
        owner_index(profile, profile.default_owner) < 0 ||
        profile.pair_access_qualified > 1 ||
        profile.owner_budget_bytes[0] == 0 || profile.owner_budget_bytes[1] == 0 ||
        !c_string_present(profile.owner_uuid[0], sizeof(profile.owner_uuid[0])) ||
        !c_string_present(profile.owner_uuid[1], sizeof(profile.owner_uuid[1]))) {
        return false;
    }
    if (profile.pair_access_qualified != 0 &&
        (!c_string_present(profile.owner_uuid[0], sizeof(profile.owner_uuid[0])) ||
         !c_string_present(profile.owner_uuid[1], sizeof(profile.owner_uuid[1])) ||
         !c_string_present(profile.qualification_id, sizeof(profile.qualification_id)))) {
        return false;
    }
    return true;
}

bool supported_allocation_prop(const CUmemAllocationProp &prop,
                               const v200_memory_pair_profile &profile) noexcept {
    if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_NONE ||
        prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        owner_index(profile, prop.location.id) < 0 ||
        prop.win32HandleMetaData != nullptr ||
        prop.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_NONE ||
        prop.allocFlags.gpuDirectRDMACapable != 0 || prop.allocFlags.usage != 0) {
        return false;
    }
    return std::all_of(std::begin(prop.allocFlags.reserved), std::end(prop.allocFlags.reserved),
                       [](unsigned char value) { return value == 0; });
}

CUmemAllocationProp owner_prop(CUdevice owner) noexcept {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = owner;
    return prop;
}

CUresult completion_error(const execution::Status &status) noexcept {
    if (status.driver_result != CUDA_SUCCESS) {
        return status.driver_result;
    }
    return status.ok() ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

} // namespace

struct MemoryManager::Impl {
    struct PhysicalExtent {
        ExtentPlan plan;
        CUmemGenericAllocationHandle handle = 0;
        CUdeviceptr mapped_base = 0;
        bool handle_live = false;
        bool mapped = false;
        bool accounted = false;
    };

    struct UseRef {
        uint64_t logical_stream_id = 0;
        execution::CompletionToken token;
        execution::CompletionProvider *provider = nullptr;
        bool retained = false;
    };

    struct Record {
        uint64_t id = 0;
        CUmemGenericAllocationHandle public_handle = 0;
        CUdeviceptr base = 0;
        size_t requested_bytes = 0;
        size_t mapped_bytes = 0;
        size_t reservation_bytes = 0;
        bool owns_reservation = false;
        bool published = false;
        bool cleanup_pending = false;
        bool allocation_api = false;
        std::vector<PhysicalExtent> extents;
        std::vector<UseRef> uses;
    };

    mutable std::mutex mutex;
    const uint32_t pair_index;
    bool configured = false;
    bool completion_fail_stopped = false;
    v200_memory_pair_profile profile{};
    v200_memory_driver_api driver{};
    execution::CompletionProvider *completion = nullptr;
    uint64_t next_record_id = 1;
    uint64_t next_handle_id = 1;
    uint64_t next_submission_id = 1;
    std::array<size_t, 2> used_bytes{{0, 0}};
    std::unordered_map<uint64_t, Record> records;
    std::unordered_map<CUmemGenericAllocationHandle, uint64_t> handles;
    std::unordered_set<uint64_t> active_submissions;
    std::condition_variable submissions_changed;
    std::map<CUdeviceptr, uint64_t> allocations_by_base;
    std::map<CUdeviceptr, size_t> reservations;

    explicit Impl(uint32_t pair) : pair_index(pair) {}

    CUresult configured_status() const noexcept {
        return configured ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
    }

    void enter_completion_fail_stop() noexcept {
        completion_fail_stopped = true;
        for (auto &item : records) {
            item.second.cleanup_pending = true;
        }
    }

    CUresult retain_prefix(std::vector<UseRef> &uses,
                           execution::CompletionProvider *provider,
                           uint64_t logical_stream_id,
                           execution::CompletionToken token) {
        auto current = uses.end();
        if (logical_stream_id != 0) {
            current = std::find_if(uses.begin(), uses.end(), [&](const UseRef &use) {
                return use.retained && use.provider == provider &&
                       use.logical_stream_id == logical_stream_id;
            });
        } else {
            current = std::find_if(uses.begin(), uses.end(), [&](const UseRef &use) {
                return use.retained && use.provider == provider &&
                       use.logical_stream_id == 0 && use.token == token;
            });
            if (current != uses.end()) {
                return CUDA_SUCCESS;
            }
        }

        if (current != uses.end() && current->token == token) {
            return CUDA_SUCCESS;
        }

        const execution::Status retained = provider->retain(token);
        if (!retained.ok()) {
            enter_completion_fail_stop();
            return completion_error(retained);
        }

        if (current != uses.end()) {
            /* The new prefix dominates the old prefix on this logical stream.
             * Retain first, then drop the old reference; a failed drop keeps
             * both references in the ledger and fail-stops all teardown. */
            const execution::Status released = provider->release(current->token);
            if (!released.ok()) {
                try {
                    uses.push_back({logical_stream_id, token, provider, true});
                } catch (const std::bad_alloc &) {
                    (void)provider->release(token);
                    enter_completion_fail_stop();
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
                enter_completion_fail_stop();
                return completion_error(released);
            }
            *current = {logical_stream_id, token, provider, true};
            return CUDA_SUCCESS;
        }

        try {
            uses.push_back({logical_stream_id, token, provider, true});
        } catch (const std::bad_alloc &) {
            const execution::Status released = provider->release(token);
            enter_completion_fail_stop();
            return released.ok() ? CUDA_ERROR_OUT_OF_MEMORY : completion_error(released);
        }
        return CUDA_SUCCESS;
    }

    CUresult granularity(CUdevice owner, size_t *out) const noexcept {
        if (out == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        CUmemAllocationProp prop = owner_prop(owner);
        size_t value = 0;
        const CUresult status = driver.mem_get_granularity(driver.user_data, &value, &prop,
                                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        if (value == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *out = value;
        return CUDA_SUCCESS;
    }

    CUresult query_both_granularities(v200_memory_pair_profile *effective) const noexcept {
        if (effective == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *effective = profile;
        for (size_t i = 0; i < 2; ++i) {
            size_t value = 0;
            const CUresult status = granularity(profile.owner_device[i], &value);
            if (status != CUDA_SUCCESS) {
                return status;
            }
            effective->owner_granularity[i] = value;
        }
        return CUDA_SUCCESS;
    }

    bool range_in_reservation(CUdeviceptr base, size_t bytes) const noexcept {
        CUdeviceptr end = 0;
        if (bytes == 0 || !checked_add_u64(base, bytes, &end)) {
            return false;
        }
        auto it = reservations.upper_bound(base);
        if (it == reservations.begin()) {
            return false;
        }
        --it;
        CUdeviceptr reservation_end = 0;
        return checked_add_u64(it->first, it->second, &reservation_end) &&
               base >= it->first && end <= reservation_end;
    }

    bool overlaps_mapped(CUdeviceptr base, size_t bytes) const noexcept {
        CUdeviceptr end = 0;
        if (!checked_add_u64(base, bytes, &end)) {
            return true;
        }
        for (const auto &item : records) {
            for (const auto &extent : item.second.extents) {
                if (!extent.mapped) {
                    continue;
                }
                CUdeviceptr extent_end = 0;
                if (!checked_add_u64(extent.mapped_base, extent.plan.size, &extent_end) ||
                    (base < extent_end && extent.mapped_base < end)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool mapped_coverage(CUdeviceptr base, size_t bytes,
                         std::vector<std::pair<uint64_t, size_t>> *covered) const {
        CUdeviceptr end = 0;
        if (bytes == 0 || !checked_add_u64(base, bytes, &end)) {
            return false;
        }
        struct Segment { CUdeviceptr begin; CUdeviceptr end; uint64_t record; size_t extent; };
        std::vector<Segment> segments;
        for (const auto &item : records) {
            for (size_t i = 0; i < item.second.extents.size(); ++i) {
                const auto &extent = item.second.extents[i];
                if (!extent.mapped) {
                    continue;
                }
                CUdeviceptr extent_end = 0;
                if (!checked_add_u64(extent.mapped_base, extent.plan.size, &extent_end)) {
                    return false;
                }
                if (base < extent_end && extent.mapped_base < end) {
                    if (extent.mapped_base < base || extent_end > end) {
                        return false;
                    }
                    segments.push_back({extent.mapped_base, extent_end, item.first, i});
                }
            }
        }
        std::sort(segments.begin(), segments.end(), [](const Segment &a, const Segment &b) {
            return a.begin < b.begin;
        });
        CUdeviceptr cursor = base;
        for (const Segment &segment : segments) {
            if (segment.begin != cursor) {
                return false;
            }
            cursor = segment.end;
            if (covered != nullptr) {
                covered->emplace_back(segment.record, segment.extent);
            }
        }
        return cursor == end;
    }

    bool mapped_range(const Record &record, CUdeviceptr base, size_t bytes) const noexcept {
        CUdeviceptr end = 0;
        if (bytes == 0 || !checked_add_u64(base, bytes, &end)) {
            return false;
        }
        CUdeviceptr cursor = base;
        while (cursor < end) {
            CUdeviceptr next = cursor;
            for (const auto &extent : record.extents) {
                if (!extent.mapped || extent.mapped_base > cursor) {
                    continue;
                }
                CUdeviceptr extent_end = 0;
                if (checked_add_u64(extent.mapped_base, extent.plan.size, &extent_end) &&
                    extent_end > next) {
                    next = extent_end;
                }
            }
            if (next == cursor) {
                return false;
            }
            cursor = std::min(next, end);
        }
        return true;
    }

    void account_if_dead(PhysicalExtent &extent) noexcept {
        if (extent.accounted && !extent.handle_live && !extent.mapped) {
            const int index = owner_index(profile, extent.plan.owner);
            if (index >= 0 && used_bytes[static_cast<size_t>(index)] >= extent.plan.size) {
                used_bytes[static_cast<size_t>(index)] -= extent.plan.size;
            }
            extent.accounted = false;
        }
    }

    bool record_has_live_resources(const Record &record) const noexcept {
        if (record.owns_reservation) {
            return true;
        }
        for (const auto &extent : record.extents) {
            if (extent.handle_live || extent.mapped || extent.accounted) {
                return true;
            }
        }
        return false;
    }

    void erase_record(uint64_t id) {
        auto found = records.find(id);
        if (found == records.end()) {
            return;
        }
        const Record &record = found->second;
        if (record.public_handle != 0) {
            handles.erase(record.public_handle);
        }
        if (record.published) {
            allocations_by_base.erase(record.base);
        }
        records.erase(found);
    }

    CUresult retain_record_uses(Record &record) {
        for (auto &use : record.uses) {
            if (!use.retained || use.provider == nullptr) {
                continue;
            }
            const execution::QueryResult query = use.provider->is_complete(use.token);
            if (!query.status.ok()) {
                return completion_error(query.status);
            }
            if (!query.complete) {
                const execution::Status waited = use.provider->wait(use.token);
                if (!waited.ok()) {
                    return completion_error(waited);
                }
            }
        }
        for (auto &use : record.uses) {
            if (!use.retained || use.provider == nullptr) {
                continue;
            }
            const execution::Status released = use.provider->release(use.token);
            if (!released.ok()) {
                return completion_error(released);
            }
            use.retained = false;
        }
        record.uses.clear();
        return CUDA_SUCCESS;
    }

    CUresult cleanup_record(uint64_t id, bool wait_for_uses) {
        auto found = records.find(id);
        if (found == records.end()) {
            return CUDA_SUCCESS;
        }
        Record &record = found->second;
        if (completion_fail_stopped) {
            record.cleanup_pending = true;
            return CUDA_ERROR_NOT_READY;
        }
        if (wait_for_uses) {
            const CUresult wait_status = retain_record_uses(record);
            if (wait_status != CUDA_SUCCESS) {
                record.cleanup_pending = true;
                return wait_status;
            }
        }
        CUresult first_error = CUDA_SUCCESS;
        for (auto &extent : record.extents) {
            if (extent.mapped) {
                const CUresult status = driver.mem_unmap(driver.user_data, extent.mapped_base,
                                                         extent.plan.size);
                if (status == CUDA_SUCCESS) {
                    extent.mapped = false;
                    account_if_dead(extent);
                } else if (first_error == CUDA_SUCCESS) {
                    first_error = status;
                }
            }
        }
        for (auto &extent : record.extents) {
            if (extent.handle_live) {
                const CUresult status = driver.mem_release(driver.user_data, extent.handle);
                if (status == CUDA_SUCCESS) {
                    extent.handle_live = false;
                    account_if_dead(extent);
                } else if (first_error == CUDA_SUCCESS) {
                    first_error = status;
                }
            }
        }
        if (record.owns_reservation) {
            bool any_mapping = false;
            for (const auto &extent : record.extents) {
                any_mapping = any_mapping || extent.mapped;
            }
            if (!any_mapping) {
                const CUresult status = driver.mem_address_free(driver.user_data, record.base,
                                                               record.reservation_bytes);
                if (status == CUDA_SUCCESS) {
                    reservations.erase(record.base);
                    record.owns_reservation = false;
                } else if (first_error == CUDA_SUCCESS) {
                    first_error = status;
                }
            }
        }
        if (first_error == CUDA_SUCCESS && !record_has_live_resources(record)) {
            erase_record(id);
            return CUDA_SUCCESS;
        }
        record.cleanup_pending = true;
        return first_error == CUDA_SUCCESS ? CUDA_ERROR_NOT_READY : first_error;
    }

    CUresult create_physical_extents(Record &record, const PlanResult &plan,
                                     const CUmemAllocationProp &requested_prop) {
        record.extents.reserve(plan.extents.size());
        for (const ExtentPlan &planned : plan.extents) {
            PhysicalExtent extent;
            extent.plan = planned;
            CUmemAllocationProp prop = requested_prop;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = planned.owner;
            CUmemGenericAllocationHandle native_handle = 0;
            const CUresult status = driver.mem_create(driver.user_data, &native_handle,
                                                     planned.size, &prop, 0);
            if (status != CUDA_SUCCESS) {
                return status;
            }
            if (native_handle == 0) {
                return CUDA_ERROR_UNKNOWN;
            }
            extent.handle = native_handle;
            extent.handle_live = true;
            const int index = owner_index(profile, planned.owner);
            if (index < 0) {
                record.extents.push_back(extent);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            const size_t owner = static_cast<size_t>(index);
            if (planned.size > std::numeric_limits<size_t>::max() - used_bytes[owner]) {
                // The native allocation exists but cannot be represented exactly in
                // this host accounting field. Keep it unaccounted and preserve it in
                // the record so cleanup remains possible; return before publication.
                record.extents.push_back(extent);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            used_bytes[owner] += planned.size;
            extent.accounted = true;
            record.extents.push_back(extent);
            if (used_bytes[owner] > profile.owner_budget_bytes[owner]) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
        return CUDA_SUCCESS;
    }

    uint64_t new_record_id() noexcept {
        if (next_record_id == 0) {
            next_record_id = 1;
        }
        return next_record_id++;
    }

    CUmemGenericAllocationHandle new_public_handle() noexcept {
        constexpr uint64_t base_tag = UINT64_C(0xF200000000000000);
        constexpr uint64_t pair_tag_bit = UINT64_C(0x0080000000000000);
        constexpr uint64_t id_mask = UINT64_C(0x007FFFFFFFFFFFFF);
        const uint64_t tag = base_tag | (pair_index == 1 ? pair_tag_bit : 0);
        for (;;) {
            const CUmemGenericAllocationHandle candidate = tag | (next_handle_id++ & id_mask);
            if (candidate != 0 && handles.find(candidate) == handles.end()) {
                return candidate;
            }
        }
    }

    Record *find_record_identity(CUdeviceptr ptr, size_t bytes = 1) {
        CUdeviceptr end = 0;
        if (bytes == 0 || !checked_add_u64(ptr, bytes, &end)) {
            return nullptr;
        }
        auto it = allocations_by_base.upper_bound(ptr);
        if (it == allocations_by_base.begin()) {
            return nullptr;
        }
        --it;
        auto record_it = records.find(it->second);
        if (record_it == records.end()) {
            return nullptr;
        }
        CUdeviceptr allocation_end = 0;
        if (!checked_add_u64(record_it->second.base, record_it->second.mapped_bytes, &allocation_end) ||
            ptr < record_it->second.base || end > allocation_end) {
            return nullptr;
        }
        return &record_it->second;
    }

    Record *find_record_for_pointer(CUdeviceptr ptr, size_t bytes = 1) {
        Record *record = find_record_identity(ptr, bytes);
        return record != nullptr && mapped_range(*record, ptr, bytes) ? record : nullptr;
    }
};

MemoryManager::MemoryManager(uint32_t pair_index) : impl_(new Impl(pair_index)) {}
MemoryManager::~MemoryManager() = default;

CUresult MemoryManager::configure(const v200_memory_pair_profile &profile,
                                  const v200_memory_driver_api &driver) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->pair_index > 1) return CUDA_ERROR_INVALID_DEVICE;
    if (!profile_valid(profile) || !driver_complete(driver)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!impl_->records.empty() || !impl_->reservations.empty()) {
        return CUDA_ERROR_NOT_READY;
    }
    impl_->profile = profile;
    impl_->driver = driver;
    impl_->configured = true;
    impl_->used_bytes = {{0, 0}};
    return CUDA_SUCCESS;
}

CUresult MemoryManager::allocate(CUdevice preferred, size_t bytes, size_t alignment,
                                 void **device_ptr) {
    if (device_ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_ptr = nullptr;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    if (bytes == 0 || (alignment != 0 && !is_power_of_two(alignment))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const int preferred_index = owner_index(impl_->profile, preferred);
    if (preferred_index < 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    v200_memory_pair_profile effective{};
    CUresult status = impl_->query_both_granularities(&effective);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    size_t round_granularity = effective.owner_granularity[static_cast<size_t>(preferred_index)];
    if (effective.pair_access_qualified != 0 &&
        !checked_lcm(effective.owner_granularity[0], effective.owner_granularity[1], &round_granularity)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    size_t mapped_bytes = 0;
    if (!checked_round_up(bytes, round_granularity, &mapped_bytes)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t used[2] = {impl_->used_bytes[0], impl_->used_bytes[1]};
    PlanResult plan = plan_extents(mapped_bytes, preferred, effective, used);
    if (plan.status != CUDA_SUCCESS) {
        return plan.status;
    }

    Impl::Record record;
    record.id = impl_->new_record_id();
    record.requested_bytes = bytes;
    record.mapped_bytes = mapped_bytes;
    record.reservation_bytes = mapped_bytes;
    record.owns_reservation = false;
    record.allocation_api = true;
    const uint64_t id = record.id;
    impl_->records.emplace(id, std::move(record));
    Impl::Record &stored = impl_->records.at(id);

    status = impl_->driver.mem_address_reserve(impl_->driver.user_data, &stored.base,
                                               mapped_bytes, alignment, 0, 0);
    if (status != CUDA_SUCCESS) {
        impl_->records.erase(id);
        return status;
    }
    CUdeviceptr reserved_end = 0;
    if (stored.base == 0 || !checked_add_u64(stored.base, mapped_bytes, &reserved_end) ||
        impl_->reservations.find(stored.base) != impl_->reservations.end()) {
        const CUresult free_status = impl_->driver.mem_address_free(impl_->driver.user_data,
                                                                   stored.base, mapped_bytes);
        if (free_status == CUDA_SUCCESS) {
            impl_->records.erase(id);
        } else {
            stored.owns_reservation = true;
            stored.reservation_bytes = mapped_bytes;
            stored.cleanup_pending = true;
        }
        return free_status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : free_status;
    }
    stored.reservation_bytes = mapped_bytes;
    stored.owns_reservation = true;
    impl_->reservations.emplace(stored.base, mapped_bytes);

    CUmemAllocationProp prop = owner_prop(preferred);
    status = impl_->create_physical_extents(stored, plan, prop);
    if (status == CUDA_SUCCESS) {
        for (auto &extent : stored.extents) {
            CUdeviceptr mapped_base = 0;
            if (!checked_add_u64(stored.base, extent.plan.logical_offset, &mapped_base)) {
                status = CUDA_ERROR_INVALID_VALUE;
                break;
            }
            status = impl_->driver.mem_map(impl_->driver.user_data, mapped_base,
                                           extent.plan.size, 0, extent.handle, 0);
            if (status != CUDA_SUCCESS) {
                break;
            }
            extent.mapped_base = mapped_base;
            extent.mapped = true;
            CUmemAccessDesc access[2]{};
            access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access[0].location.id = preferred;
            access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            size_t access_count = 1;
            if (impl_->profile.pair_access_qualified != 0) {
                access[0].location.id = impl_->profile.owner_device[0];
                access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access[1].location.id = impl_->profile.owner_device[1];
                access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                access_count = 2;
            }
            status = impl_->driver.mem_set_access(impl_->driver.user_data, mapped_base,
                                                  extent.plan.size, access, access_count);
            if (status != CUDA_SUCCESS) {
                break;
            }
        }
    }
    if (status != CUDA_SUCCESS) {
        (void)impl_->cleanup_record(id, false);
        return status;
    }
    stored.published = true;
    impl_->allocations_by_base.emplace(stored.base, id);
    *device_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(stored.base));
    return CUDA_SUCCESS;
}

CUresult MemoryManager::release(void *device_ptr) {
    if (device_ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUdeviceptr base = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(device_ptr));
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    impl_->submissions_changed.wait(lock, [&] { return impl_->active_submissions.empty(); });
    auto it = impl_->allocations_by_base.find(base);
    if (it == impl_->allocations_by_base.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const uint64_t id = it->second;
    const auto record = impl_->records.find(id);
    if (record == impl_->records.end() || !record->second.allocation_api) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUresult status = impl_->cleanup_record(id, true);
    return status;
}

CUresult MemoryManager::query(const void *ptr, v200_allocation_info *out) const {
    if (ptr == nullptr || out == nullptr || out->struct_size < sizeof(v200_allocation_info)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUdeviceptr address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    Impl::Record *record = impl_->find_record_for_pointer(address);
    if (record == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out->abi_version = V200_MEMORY_ABI_VERSION;
    out->allocation_id = record->id;
    out->base = record->base;
    out->requested_bytes = record->requested_bytes;
    out->mapped_bytes = record->mapped_bytes;
    out->owner_bytes[0] = 0;
    out->owner_bytes[1] = 0;
    out->extent_count = static_cast<uint32_t>(record->extents.size());
    uint32_t flags = record->allocation_api
        ? V200_ALLOCATION_INFO_FLAG_ALLOCATION_API : 0u;
    if (record->published && !record->extents.empty()) {
        const int first_owner = owner_index(impl_->profile, record->extents[0].plan.owner);
        if (first_owner >= 0) {
            flags |= V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_KNOWN;
            if (first_owner == 1) {
                flags |= V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_1;
            }
        }
    }
    out->flags = flags;
    for (const auto &extent : record->extents) {
        const int index = owner_index(impl_->profile, extent.plan.owner);
        if (index >= 0) {
            out->owner_bytes[static_cast<size_t>(index)] += extent.plan.size;
        }
    }
    return CUDA_SUCCESS;
}

bool MemoryManager::is_configured() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->configured;
}

bool MemoryManager::owns_pointer(const void *ptr) const {
    if (ptr == nullptr) {
        return false;
    }
    const CUdeviceptr address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->configured) {
        return false;
    }
    auto it = impl_->allocations_by_base.upper_bound(address);
    if (it == impl_->allocations_by_base.begin()) {
        return false;
    }
    --it;
    auto record = impl_->records.find(it->second);
    CUdeviceptr end = 0;
    return record != impl_->records.end() &&
           checked_add_u64(record->second.base, record->second.mapped_bytes, &end) &&
           address >= record->second.base && address < end;
}

bool MemoryManager::owns_pointer_range(const void *ptr, size_t size) const {
    if (ptr == nullptr || size == 0) return false;
    const CUdeviceptr address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->configured && impl_->find_record_for_pointer(address, size) != nullptr;
}

bool MemoryManager::owns_reservation_range(CUdeviceptr ptr, size_t size) const {
    if (size == 0) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->configured && impl_->range_in_reservation(ptr, size);
}

CUresult MemoryManager::get_profile(v200_memory_pair_profile *out) const {
    if (out == nullptr || out->struct_size < sizeof(v200_memory_pair_profile)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    *out = impl_->profile;
    return CUDA_SUCCESS;
}

CUresult MemoryManager::get_usage(v200_memory_usage_info *out) const {
    if (out == nullptr || out->struct_size < sizeof(v200_memory_usage_info)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    out->abi_version = V200_MEMORY_ABI_VERSION;
    out->owner_used_bytes[0] = impl_->used_bytes[0];
    out->owner_used_bytes[1] = impl_->used_bytes[1];
    out->owner_budget_bytes[0] = impl_->profile.owner_budget_bytes[0];
    out->owner_budget_bytes[1] = impl_->profile.owner_budget_bytes[1];
    out->live_allocation_count = 0;
    out->live_reservation_count = impl_->reservations.size();
    out->cleanup_pending_count = 0;
    for (const auto &item : impl_->records) {
        if (item.second.published) {
            ++out->live_allocation_count;
        }
        if (item.second.cleanup_pending) {
            ++out->cleanup_pending_count;
        }
    }
    return CUDA_SUCCESS;
}

CUresult MemoryManager::vmm_get_granularity(size_t *granularity,
                                            const CUmemAllocationProp *prop,
                                            CUmemAllocationGranularity_flags option) {
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (!supported_allocation_prop(*prop, impl_->profile)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM &&
        option != CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return impl_->driver.mem_get_granularity(impl_->driver.user_data, granularity, prop, option);
}

CUresult MemoryManager::vmm_address_reserve(CUdeviceptr *ptr, size_t size,
                                            size_t alignment, CUdeviceptr requested,
                                            unsigned long long flags) {
    if (ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ptr = 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (size == 0 || requested != 0 || flags != 0 ||
        (alignment != 0 && !is_power_of_two(alignment))) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUdeviceptr base = 0;
    const CUresult status = impl_->driver.mem_address_reserve(impl_->driver.user_data, &base,
                                                              size, alignment, 0, 0);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    CUdeviceptr end = 0;
    if (base == 0 || !checked_add_u64(base, size, &end) ||
        impl_->reservations.find(base) != impl_->reservations.end()) {
        const CUresult free_status = impl_->driver.mem_address_free(impl_->driver.user_data, base, size);
        return free_status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : free_status;
    }
    impl_->reservations.emplace(base, size);
    *ptr = base;
    return CUDA_SUCCESS;
}

CUresult MemoryManager::vmm_address_free(CUdeviceptr ptr, size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    auto reservation = impl_->reservations.find(ptr);
    if (reservation == impl_->reservations.end() || reservation->second != size) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (const auto &record : impl_->records) {
        for (const auto &extent : record.second.extents) {
            if (extent.mapped && extent.mapped_base >= ptr &&
                extent.mapped_base - ptr < size) {
                return CUDA_ERROR_NOT_READY;
            }
        }
    }
    const CUresult status = impl_->driver.mem_address_free(impl_->driver.user_data, ptr, size);
    if (status == CUDA_SUCCESS) {
        impl_->reservations.erase(reservation);
    }
    return status;
}

CUresult MemoryManager::vmm_create(CUmemGenericAllocationHandle *handle, size_t size,
                                   const CUmemAllocationProp *prop,
                                   unsigned long long flags) {
    if (handle == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (size == 0 || flags != 0 || !supported_allocation_prop(*prop, impl_->profile)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    v200_memory_pair_profile effective{};
    CUresult status = impl_->query_both_granularities(&effective);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    const int preferred_index = owner_index(impl_->profile, prop->location.id);
    size_t preferred_granularity = 0;
    status = impl_->granularity(prop->location.id, &preferred_granularity);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    if (size % preferred_granularity != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    effective.owner_granularity[static_cast<size_t>(preferred_index)] = preferred_granularity;
    const size_t used[2] = {impl_->used_bytes[0], impl_->used_bytes[1]};
    PlanResult plan = plan_extents(size, prop->location.id, effective, used);
    if (plan.status != CUDA_SUCCESS) {
        return plan.status;
    }

    Impl::Record record;
    record.id = impl_->new_record_id();
    record.requested_bytes = size;
    record.mapped_bytes = size;
    record.allocation_api = false;
    const uint64_t id = record.id;
    impl_->records.emplace(id, std::move(record));
    Impl::Record &stored = impl_->records.at(id);
    status = impl_->create_physical_extents(stored, plan, *prop);
    if (status != CUDA_SUCCESS) {
        stored.cleanup_pending = true;
        (void)impl_->cleanup_record(id, false);
        return status;
    }
    stored.public_handle = impl_->new_public_handle();
    impl_->handles.emplace(stored.public_handle, id);
    *handle = stored.public_handle;
    return CUDA_SUCCESS;
}

CUresult MemoryManager::vmm_map(CUdeviceptr ptr, size_t size, size_t offset,
                                CUmemGenericAllocationHandle handle,
                                unsigned long long flags) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    auto handle_it = impl_->handles.find(handle);
    if (handle_it == impl_->handles.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    Impl::Record &record = impl_->records.at(handle_it->second);
    if (std::any_of(record.extents.begin(), record.extents.end(),
                    [](const Impl::PhysicalExtent &extent) { return extent.mapped; }) ||
        std::any_of(record.extents.begin(), record.extents.end(),
                    [](const Impl::PhysicalExtent &extent) { return !extent.handle_live; })) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (offset != 0 || flags != 0 || size != record.requested_bytes ||
        record.extents.empty() || !impl_->range_in_reservation(ptr, size) ||
        impl_->overlaps_mapped(ptr, size)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    std::vector<size_t> newly_mapped;
    for (size_t i = 0; i < record.extents.size(); ++i) {
        auto &extent = record.extents[i];
        CUdeviceptr mapped_base = 0;
        if (!checked_add_u64(ptr, extent.plan.logical_offset, &mapped_base)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const CUresult status = impl_->driver.mem_map(impl_->driver.user_data, mapped_base,
                                                      extent.plan.size, 0, extent.handle, 0);
        if (status != CUDA_SUCCESS) {
            CUresult rollback_error = CUDA_SUCCESS;
            for (auto reverse = newly_mapped.rbegin(); reverse != newly_mapped.rend(); ++reverse) {
                auto &previous = record.extents[*reverse];
                const CUresult unmap_status = impl_->driver.mem_unmap(impl_->driver.user_data,
                                                                      previous.mapped_base,
                                                                      previous.plan.size);
                if (unmap_status == CUDA_SUCCESS) {
                    previous.mapped = false;
                    impl_->account_if_dead(previous);
                } else if (rollback_error == CUDA_SUCCESS) {
                    rollback_error = unmap_status;
                }
            }
            if (rollback_error != CUDA_SUCCESS) {
                record.base = ptr;
                record.cleanup_pending = true;
                return rollback_error;
            }
            return status;
        }
        extent.mapped_base = mapped_base;
        extent.mapped = true;
        newly_mapped.push_back(i);
    }
    record.base = ptr;
    record.published = true;
    impl_->allocations_by_base.emplace(ptr, record.id);
    return CUDA_SUCCESS;
}

CUresult MemoryManager::vmm_set_access(CUdeviceptr ptr, size_t size,
                                       const CUmemAccessDesc *desc, size_t count) {
    if (desc == nullptr || count == 0 || count > 2) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    for (size_t i = 0; i < count; ++i) {
        if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
            owner_index(impl_->profile, desc[i].location.id) < 0 ||
            (desc[i].flags != CU_MEM_ACCESS_FLAGS_PROT_NONE &&
             desc[i].flags != CU_MEM_ACCESS_FLAGS_PROT_READ &&
             desc[i].flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        if (i != 0 && desc[i].location.id == desc[0].location.id) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (desc[i].location.id != impl_->profile.owner_device[0] &&
            desc[i].location.id != impl_->profile.owner_device[1]) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    std::vector<std::pair<uint64_t, size_t>> covered;
    if (!impl_->mapped_coverage(ptr, size, &covered)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (impl_->profile.pair_access_qualified == 0) {
        for (const auto &item : covered) {
            const auto found = impl_->records.find(item.first);
            if (found == impl_->records.end()) {
                return CUDA_ERROR_UNKNOWN;
            }
            const CUdevice extent_owner = found->second.extents[item.second].plan.owner;
            for (size_t i = 0; i < count; ++i) {
                if (desc[i].location.id != extent_owner) {
                    return CUDA_ERROR_NOT_SUPPORTED;
                }
            }
        }
    }
    // One logical-device permission applies to both compute domains, including
    // revocation; a peer launch must never bypass a logical PROT_NONE.
    if (impl_->profile.pair_access_qualified != 0 && count == 1) {
        CUmemAccessDesc pair_access[2]{};
        pair_access[0] = desc[0];
        pair_access[1] = desc[0];
        pair_access[0].location.id = impl_->profile.owner_device[0];
        pair_access[1].location.id = impl_->profile.owner_device[1];
        return impl_->driver.mem_set_access(impl_->driver.user_data, ptr, size,
                                            pair_access, 2);
    }
    return impl_->driver.mem_set_access(impl_->driver.user_data, ptr, size, desc, count);
}

CUresult MemoryManager::vmm_unmap(CUdeviceptr ptr, size_t size) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    impl_->submissions_changed.wait(lock, [&] { return impl_->active_submissions.empty(); });
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    std::vector<std::pair<uint64_t, size_t>> covered;
    if (!impl_->mapped_coverage(ptr, size, &covered) || covered.empty()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (const auto &item : covered) {
        auto found = impl_->records.find(item.first);
        if (found == impl_->records.end()) {
            return CUDA_ERROR_UNKNOWN;
        }
        const CUresult status = impl_->retain_record_uses(found->second);
        if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    const CUresult status = impl_->driver.mem_unmap(impl_->driver.user_data, ptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    for (const auto &item : covered) {
        auto found = impl_->records.find(item.first);
        if (found == impl_->records.end()) {
            continue;
        }
        Impl::Record &record = found->second;
        Impl::PhysicalExtent &extent = record.extents[item.second];
        extent.mapped = false;
        impl_->account_if_dead(extent);
    }
    std::vector<uint64_t> now_unmapped;
    for (const auto &item : covered) {
        if (std::find(now_unmapped.begin(), now_unmapped.end(), item.first) != now_unmapped.end()) {
            continue;
        }
        now_unmapped.push_back(item.first);
        auto found = impl_->records.find(item.first);
        if (found == impl_->records.end()) {
            continue;
        }
        const bool any_mapping = std::any_of(found->second.extents.begin(), found->second.extents.end(),
                                             [](const Impl::PhysicalExtent &extent) { return extent.mapped; });
        if (!any_mapping && found->second.published) {
            impl_->allocations_by_base.erase(found->second.base);
            found->second.published = false;
        }
    }
    std::vector<uint64_t> dead;
    for (const auto &item : covered) {
        auto found = impl_->records.find(item.first);
        if (found != impl_->records.end() && !impl_->record_has_live_resources(found->second)) {
            dead.push_back(item.first);
        }
    }
    std::sort(dead.begin(), dead.end());
    dead.erase(std::unique(dead.begin(), dead.end()), dead.end());
    for (uint64_t id : dead) {
        impl_->erase_record(id);
    }
    return CUDA_SUCCESS;
}

CUresult MemoryManager::vmm_release(CUmemGenericAllocationHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    auto handle_it = impl_->handles.find(handle);
    if (handle_it == impl_->handles.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const uint64_t id = handle_it->second;
    auto record_it = impl_->records.find(id);
    if (record_it == impl_->records.end()) {
        impl_->handles.erase(handle_it);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    Impl::Record &record = record_it->second;
    CUresult first_error = CUDA_SUCCESS;
    for (auto &extent : record.extents) {
        if (!extent.handle_live) {
            continue;
        }
        const CUresult status = impl_->driver.mem_release(impl_->driver.user_data, extent.handle);
        if (status == CUDA_SUCCESS) {
            extent.handle_live = false;
            impl_->account_if_dead(extent);
        } else if (first_error == CUDA_SUCCESS) {
            first_error = status;
        }
    }
    const bool any_live_handle = std::any_of(record.extents.begin(), record.extents.end(),
                                             [](const Impl::PhysicalExtent &extent) { return extent.handle_live; });
    if (!any_live_handle) {
        impl_->handles.erase(handle_it);
        record.public_handle = 0;
    }
    if (first_error == CUDA_SUCCESS && !impl_->record_has_live_resources(record)) {
        impl_->erase_record(id);
    }
    return first_error;
}

bool MemoryManager::owns_handle(CUmemGenericAllocationHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->handles.find(handle) != impl_->handles.end();
}

void MemoryManager::set_completion_provider(execution::CompletionProvider *provider) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->completion = provider;
}

CUresult MemoryManager::retain_use(CUdeviceptr ptr, uint64_t logical_stream_id,
                                  execution::CompletionToken token) {
    if (!token.valid()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    Impl::Record *record = impl_->find_record_for_pointer(ptr);
    if (record == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (impl_->completion == nullptr) {
        impl_->enter_completion_fail_stop();
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return impl_->retain_prefix(record->uses, impl_->completion,
                                logical_stream_id, token);
}

CUresult MemoryManager::retain_all_use(uint64_t logical_stream_id,
                                      execution::CompletionToken token) {
    if (!token.valid()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    for (const auto &item : impl_->records) {
        if (item.second.cleanup_pending) {
            return CUDA_ERROR_NOT_READY;
        }
    }
    bool has_live_mapping = false;
    for (const auto &item : impl_->records) {
        has_live_mapping = has_live_mapping || item.second.published;
    }
    if (!has_live_mapping) {
        return CUDA_SUCCESS;
    }
    if (impl_->completion == nullptr) {
        impl_->enter_completion_fail_stop();
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    for (auto &item : impl_->records) {
        Impl::Record &record = item.second;
        if (!record.published) {
            continue;
        }
        const CUresult status = impl_->retain_prefix(record.uses, impl_->completion,
                                                     logical_stream_id, token);
        if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    return CUDA_SUCCESS;
}

CUresult MemoryManager::begin_submission(uint64_t *lease) {
    if (lease == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *lease = 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->configured) {
        return CUDA_SUCCESS;
    }
    if (impl_->completion_fail_stopped) {
        return CUDA_ERROR_NOT_READY;
    }
    uint64_t candidate = impl_->next_submission_id++;
    while (candidate == 0 || impl_->active_submissions.find(candidate) != impl_->active_submissions.end()) {
        candidate = impl_->next_submission_id++;
    }
    try {
        impl_->active_submissions.insert(candidate);
    } catch (const std::bad_alloc &) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *lease = candidate;
    return CUDA_SUCCESS;
}

CUresult MemoryManager::end_submission(uint64_t lease, bool completion_recorded) {
    if (lease == 0) {
        return CUDA_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->active_submissions.find(lease);
    if (found == impl_->active_submissions.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    bool has_live_mapping = false;
    for (const auto &item : impl_->records) {
        has_live_mapping = has_live_mapping || item.second.published;
    }
    CUresult result = CUDA_SUCCESS;
    if (!completion_recorded && has_live_mapping) {
        impl_->enter_completion_fail_stop();
        result = CUDA_ERROR_NOT_READY;
    }
    impl_->active_submissions.erase(found);
    impl_->submissions_changed.notify_all();
    return result;
}

CUresult MemoryManager::retry_cleanup() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->configured_status() != CUDA_SUCCESS) {
        return impl_->configured_status();
    }
    impl_->submissions_changed.wait(lock, [&] { return impl_->active_submissions.empty(); });
    std::vector<uint64_t> pending;
    for (const auto &item : impl_->records) {
        if (item.second.cleanup_pending) {
            pending.push_back(item.first);
        }
    }
    CUresult first_error = CUDA_SUCCESS;
    for (uint64_t id : pending) {
        const CUresult status = impl_->cleanup_record(id, true);
        if (status != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
            first_error = status;
        }
    }
    return first_error;
}

MemoryManager *manager_for_pair(uint32_t pair_index) noexcept {
    if (pair_index > 1) return nullptr;
    static std::array<MemoryManager, 2> managers{{MemoryManager(0), MemoryManager(1)}};
    return &managers[pair_index];
}

MemoryManager &default_manager() {
    return *manager_for_pair(0);
}

namespace {

struct OwnerRoute {
    MemoryManager *manager = nullptr;
    bool found = false;
    bool ambiguous = false;
};

template <typename Predicate>
OwnerRoute find_unique_owner(Predicate predicate) {
    OwnerRoute route;
    for (uint32_t i = 0; i != 2; ++i) {
        MemoryManager *candidate = manager_for_pair(i);
        if (candidate != nullptr && predicate(*candidate)) {
            if (route.found) {
                route.manager = nullptr;
                route.ambiguous = true;
                return route;
            }
            route.manager = candidate;
            route.found = true;
        }
    }
    return route;
}

OwnerRoute pointer_owner(const void *ptr) {
    return find_unique_owner([&](const MemoryManager &manager) {
        return manager.owns_pointer(ptr);
    });
}

OwnerRoute reservation_owner(CUdeviceptr ptr, size_t size) {
    return find_unique_owner([&](const MemoryManager &manager) {
        return manager.owns_reservation_range(ptr, size);
    });
}

OwnerRoute handle_owner(CUmemGenericAllocationHandle handle) {
    return find_unique_owner([&](const MemoryManager &manager) {
        return manager.owns_handle(handle);
    });
}

MemoryManager &selected_manager() {
    return *manager_for_pair(selected_pair_index);
}

} // namespace

} // namespace v200::memory

extern "C" {

CUresult v200_memory_configure(const v200_memory_pair_profile *profile,
                               const v200_memory_driver_api *driver) {
    if (profile == nullptr || driver == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return v200::memory::selected_manager().configure(*profile, *driver);
}

CUresult v200_memory_select_pair(uint32_t pair_index) {
    if (pair_index > 1) return CUDA_ERROR_INVALID_DEVICE;
    v200::memory::selected_pair_index = pair_index;
    return CUDA_SUCCESS;
}

uint32_t v200_memory_current_pair(void) {
    return v200::memory::selected_pair_index;
}

CUresult v200_memory_allocate_on(CUdevice preferred_device, size_t bytes,
                                 size_t alignment, void **device_ptr) {
    return v200::memory::selected_manager().allocate(preferred_device, bytes, alignment, device_ptr);
}

CUresult v200_memory_allocate(size_t bytes, size_t alignment, void **device_ptr) {
    v200_memory_pair_profile profile{};
    profile.struct_size = sizeof(profile);
    const CUresult status = v200_memory_get_profile(&profile);
    if (status != CUDA_SUCCESS) {
        if (device_ptr != nullptr) {
            *device_ptr = nullptr;
        }
        return status;
    }
    return v200::memory::selected_manager().allocate(profile.default_owner, bytes, alignment, device_ptr);
}

int v200_memory_is_configured(void) {
    return v200::memory::selected_manager().is_configured() ? 1 : 0;
}

CUresult v200_memory_get_profile(v200_memory_pair_profile *out) {
    return v200::memory::selected_manager().get_profile(out);
}

CUresult v200_memory_get_profile_for_pair(uint32_t pair_index,
                                          v200_memory_pair_profile *out) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager == nullptr ? CUDA_ERROR_INVALID_DEVICE : manager->get_profile(out);
}

CUresult v200_memory_release(void *device_ptr) {
    const auto route = v200::memory::pointer_owner(device_ptr);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->release(device_ptr);
}

CUresult v200_memory_query(const void *ptr, v200_allocation_info *out) {
    const auto route = v200::memory::pointer_owner(ptr);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->query(ptr, out);
}

int v200_memory_owns_pointer(const void *ptr) {
    const auto route = v200::memory::pointer_owner(ptr);
    return route.found || route.ambiguous ? 1 : 0;
}

CUresult v200_memory_query_for_pair(uint32_t pair_index, const void *ptr,
                                    v200_allocation_info *out) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager == nullptr ? CUDA_ERROR_INVALID_DEVICE : manager->query(ptr, out);
}

int v200_memory_owns_pointer_for_pair(uint32_t pair_index, const void *ptr) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager != nullptr && manager->owns_pointer(ptr) ? 1 : 0;
}

int v200_memory_owns_range_for_pair(uint32_t pair_index, const void *ptr,
                                    size_t size) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager != nullptr && manager->owns_pointer_range(ptr, size) ? 1 : 0;
}

CUresult v200_memory_retain_use(const void *ptr, uint64_t logical_stream_id,
                               uint64_t token_slot, uint64_t token_generation) {
    if (ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    const auto route = v200::memory::pointer_owner(ptr);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    if (route.found && route.manager != &v200::memory::selected_manager()) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return (route.found ? route.manager : &v200::memory::selected_manager())->retain_use(
        address, logical_stream_id, {token_slot, token_generation});
}

CUresult v200_memory_retain_use_for_pair(uint32_t pair_index, const void *ptr,
                                         uint64_t logical_stream_id,
                                         uint64_t token_slot,
                                         uint64_t token_generation) {
    if (ptr == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto *manager = v200::memory::manager_for_pair(pair_index);
    if (manager == nullptr) return CUDA_ERROR_INVALID_DEVICE;
    const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    return manager->retain_use(address, logical_stream_id,
                               {token_slot, token_generation});
}

CUresult v200_memory_retain_all_use(uint64_t logical_stream_id,
                                    uint64_t token_slot,
                                    uint64_t token_generation) {
    return v200::memory::selected_manager().retain_all_use(
        logical_stream_id, {token_slot, token_generation});
}

CUresult v200_memory_begin_submission(uint64_t *lease) {
    return v200::memory::selected_manager().begin_submission(lease);
}

CUresult v200_memory_end_submission(uint64_t lease, int completion_recorded) {
    if (completion_recorded != 0 && completion_recorded != 1) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return v200::memory::selected_manager().end_submission(lease,
                                                          completion_recorded != 0);
}

CUresult v200_memory_begin_submission_for_pair(uint32_t pair_index,
                                               uint64_t *lease) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager == nullptr ? CUDA_ERROR_INVALID_DEVICE : manager->begin_submission(lease);
}

CUresult v200_memory_end_submission_for_pair(uint32_t pair_index,
                                             uint64_t lease,
                                             int completion_recorded) {
    if (completion_recorded != 0 && completion_recorded != 1) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager == nullptr ? CUDA_ERROR_INVALID_DEVICE
                              : manager->end_submission(lease, completion_recorded != 0);
}

CUresult v200_memory_get_usage(v200_memory_usage_info *out) {
    return v200::memory::selected_manager().get_usage(out);
}

CUresult v200_memory_get_usage_for_pair(uint32_t pair_index,
                                        v200_memory_usage_info *out) {
    auto *manager = v200::memory::manager_for_pair(pair_index);
    return manager == nullptr ? CUDA_ERROR_INVALID_DEVICE : manager->get_usage(out);
}

CUresult v200_memory_vmm_get_granularity(size_t *granularity,
                                         const CUmemAllocationProp *prop,
                                         CUmemAllocationGranularity_flags option) {
    return v200::memory::selected_manager().vmm_get_granularity(granularity, prop, option);
}

CUresult v200_memory_vmm_address_reserve(CUdeviceptr *ptr, size_t size,
                                         size_t alignment, CUdeviceptr requested,
                                         unsigned long long flags) {
    return v200::memory::selected_manager().vmm_address_reserve(ptr, size, alignment, requested, flags);
}

CUresult v200_memory_vmm_address_free(CUdeviceptr ptr, size_t size) {
    const auto route = v200::memory::reservation_owner(ptr, size);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->vmm_address_free(ptr, size);
}

CUresult v200_memory_vmm_create(CUmemGenericAllocationHandle *handle, size_t size,
                                const CUmemAllocationProp *prop, unsigned long long flags) {
    return v200::memory::selected_manager().vmm_create(handle, size, prop, flags);
}

CUresult v200_memory_vmm_map(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle, unsigned long long flags) {
    const auto reservation = v200::memory::reservation_owner(ptr, size);
    const auto allocation = v200::memory::handle_owner(handle);
    if (reservation.ambiguous) {
        if (!allocation.found) return CUDA_ERROR_INVALID_DEVICE;
        return allocation.manager->vmm_map(ptr, size, offset, handle, flags);
    }
    if (reservation.found && allocation.found && reservation.manager != allocation.manager) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if (allocation.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    auto *manager = allocation.found ? allocation.manager
                   : reservation.found ? reservation.manager
                   : &v200::memory::selected_manager();
    return manager->vmm_map(ptr, size, offset, handle, flags);
}

CUresult v200_memory_vmm_set_access(CUdeviceptr ptr, size_t size,
                                    const CUmemAccessDesc *desc, size_t count) {
    const auto route = v200::memory::reservation_owner(ptr, size);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->vmm_set_access(ptr, size, desc, count);
}

CUresult v200_memory_vmm_unmap(CUdeviceptr ptr, size_t size) {
    const auto route = v200::memory::reservation_owner(ptr, size);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->vmm_unmap(ptr, size);
}

CUresult v200_memory_vmm_release(CUmemGenericAllocationHandle handle) {
    const auto route = v200::memory::handle_owner(handle);
    if (route.ambiguous) return CUDA_ERROR_INVALID_DEVICE;
    return (route.found ? route.manager : &v200::memory::selected_manager())->vmm_release(handle);
}

int v200_memory_owns_handle(CUmemGenericAllocationHandle handle) {
    return v200::memory::handle_owner(handle).found ? 1 : 0;
}

CUresult v200_memory_retry_cleanup(void) {
    return v200::memory::selected_manager().retry_cleanup();
}

} // extern "C"
