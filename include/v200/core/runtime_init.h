#ifndef V200_CORE_RUNTIME_INIT_H
#define V200_CORE_RUNTIME_INIT_H

#include <cuda.h>

#if defined(__GNUC__)
#define V200_CORE_API __attribute__((visibility("default")))
#else
#define V200_CORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Lazily initializes the exact qualified V200 pair. This function performs
 * CUDA work only when called by the pair-mode frontdoor, never at DSO load. */
V200_CORE_API CUresult CUDAAPI v200_initialize_profile(void);

#ifdef __cplusplus
}
#endif

#endif
