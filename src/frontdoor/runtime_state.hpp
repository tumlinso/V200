#pragma once

#include "v200/frontdoor/runtime.h"
#include "v200/execution/owner_execution_c.h"

#include <cstdint>

namespace v200::frontdoor::detail {

class InternalCallScope {
public:
    InternalCallScope() noexcept;
    ~InternalCallScope();
    InternalCallScope(const InternalCallScope &) = delete;
    InternalCallScope &operator=(const InternalCallScope &) = delete;
};

bool internal_call() noexcept;
uint32_t current_mode() noexcept;
bool pair_mode() noexcept;
bool presentation_enabled() noexcept;
uint64_t stream_id(const void *stream) noexcept;
void forget_stream(const void *stream) noexcept;

template <class Function>
Function resolve_next(const char *symbol) noexcept {
    /* Prefer normal ELF interposition, then handle plugin-local CUDA DSOs. */
    extern void *lookup_native(const char *) noexcept;
    return reinterpret_cast<Function>(lookup_native(symbol));
}

void *lookup_v200(const char *symbol) noexcept;
void *lookup_native(const char *symbol) noexcept;
void *lookup_driver(const char *symbol) noexcept;
void *lookup_runtime(const char *symbol) noexcept;
const char *mode_name(uint32_t mode) noexcept;

v200_execution_handle *execution_handle() noexcept;
v200_logical_device_id execution_device() noexcept;
v200_owner_id execution_owner() noexcept;
int register_execution_stream(void *stream) noexcept;
int observe_execution_submission(void *stream, int driver_result) noexcept;
int execution_token_for_stream(void *stream, v200_completion_token *token) noexcept;
int execution_token_for_device(v200_completion_token *token) noexcept;
int execution_wait_token(v200_completion_token token, int *driver_result) noexcept;
int execution_release_token(v200_completion_token token) noexcept;
int attach_execution(v200_execution_handle *handle, v200_logical_device_id device,
                     v200_owner_id owner) noexcept;
void detach_execution(v200_execution_handle *handle) noexcept;

} // namespace v200::frontdoor::detail
