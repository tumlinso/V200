#ifndef V200_CORE_RUNTIME_INIT_H
#define V200_CORE_RUNTIME_INIT_H

#include <cuda.h>
#include <stdint.h>

#if defined(__GNUC__)
#define V200_CORE_API __attribute__((visibility("default")))
#else
#define V200_CORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Lazily initializes logical pair 0 for backwards compatibility. This
 * function performs CUDA work only when called by the pair-mode frontdoor,
 * never at DSO load. */
V200_CORE_API CUresult CUDAAPI v200_initialize_profile(void);
/* Lazily initializes one qualified pair (indices 0 or 1). */
V200_CORE_API CUresult CUDAAPI v200_initialize_profile_pair(uint32_t pair_index);

#ifdef __cplusplus
}
#endif

#endif
