#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <link.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>

extern "C" char **environ;

namespace {

bool starts_with(const char *s, const char *prefix) noexcept;

bool is_cuda_symbol(const char *name) noexcept {
    if (name == nullptr) return false;
    return starts_with(name, "cu") || starts_with(name, "cuda") ||
           starts_with(name, "cublas") || starts_with(name, "nccl");
}

bool starts_with(const char *s, const char *prefix) noexcept {
    if (s == nullptr) return false;
    while (*prefix != '\0') {
        if (*s++ != *prefix++) return false;
    }
    return true;
}

const char *classification(const char *name) noexcept {
    static const char *const wrapped[] = {
        "cuDeviceGet", "cuDeviceGetCount", "cuGetProcAddress", "cuMemAddressReserve",
        "cuMemAddressFree", "cuMemCreate", "cuMemGetAllocationGranularity", "cuMemMap",
        "cuMemRelease", "cuMemSetAccess", "cuMemUnmap", "cuLaunchKernel",
        "cuMemcpyDtoDAsync", "cuMemcpyDtoHAsync", "cuMemcpyHtoDAsync", "cuStreamSynchronize",
        "cuStreamWaitEvent", "cudaDeviceCanAccessPeer", "cudaDeviceEnablePeerAccess",
        "cudaDeviceGetAttribute", "cudaDeviceSynchronize", "cudaEventRecord",
        "cudaEventSynchronize", "cudaFree", "cudaGetDevice", "cudaGetDeviceCount",
        "cudaGetDeviceProperties", "cudaGetDriverEntryPoint", "cudaMalloc", "cudaMemGetInfo",
        "cudaMemcpy2DAsync", "cudaMemcpyAsync", "cudaMemcpyPeerAsync", "cudaMemsetAsync",
        "cudaSetDevice", "cudaStreamCreateWithFlags", "cudaStreamDestroy",
        "cudaStreamSynchronize", "cudaStreamWaitEvent", "cudaLaunchKernel",
        "cudaGraphLaunch"
    };
    for (const char *candidate : wrapped) {
        if (starts_with(name, candidate)) return "wrapper_candidate";
    }
    if (starts_with(name, "cublas") || starts_with(name, "nccl")) {
        return "native_library_unqualified";
    }
    return "unclassified";
}

bool log_fd(int &fd) noexcept {
    static int cached_fd = -2;
    if (cached_fd != -2) {
        fd = cached_fd;
        return fd >= 0;
    }
    cached_fd = -1;
    for (char **item = ::environ; item != nullptr && *item != nullptr; ++item) {
        constexpr char key[] = "V200_ABI_AUDIT_FD=";
        const char *value = *item;
        std::size_t i = 0;
        while (key[i] != '\0' && value[i] == key[i]) ++i;
        if (key[i] != '\0') continue;
        int parsed = 0;
        for (; value[i] >= '0' && value[i] <= '9'; ++i) parsed = parsed * 10 + value[i] - '0';
        if (value[i] == '\0') cached_fd = parsed;
        break;
    }
    fd = cached_fd;
    return fd >= 0;
}

void append(char *out, std::size_t capacity, std::size_t &used, const char *value) noexcept {
    if (value == nullptr) value = "";
    for (; *value != '\0' && used + 1 < capacity; ++value) {
        const char c = (*value == '\t' || *value == '\n' || *value == '\r') ? '_' : *value;
        out[used++] = c;
    }
}

void append_hex(char *out, std::size_t capacity, std::size_t &used, std::uintptr_t value) noexcept {
    char digits[2 * sizeof(value)];
    std::size_t count = 0;
    do {
        const unsigned digit = static_cast<unsigned>(value & 0xfu);
        digits[count++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value >>= 4;
    } while (value != 0 && count < sizeof(digits));
    while (count > 0 && used + 1 < capacity) out[used++] = digits[--count];
}

void record(const char *name, const link_map *from, const link_map *to,
            std::uintptr_t address) noexcept {
    int fd = -1;
    if (!log_fd(fd) || !is_cuda_symbol(name)) return;
    char line[1024];
    std::size_t used = 0;
    append(line, sizeof(line), used, classification(name));
    append(line, sizeof(line), used, "\t");
    append(line, sizeof(line), used, name);
    append(line, sizeof(line), used, "\t");
    append(line, sizeof(line), used, from ? from->l_name : "");
    append(line, sizeof(line), used, "\t");
    append(line, sizeof(line), used, to ? to->l_name : "");
    append(line, sizeof(line), used, "\t0x");
    append_hex(line, sizeof(line), used, address);
    append(line, sizeof(line), used, "\n");
    (void)syscall(SYS_write, fd, line, used);
}

} // namespace

extern "C" unsigned int la_version(unsigned int version) {
    return version >= LAV_CURRENT ? LAV_CURRENT : version;
}

extern "C" unsigned int la_objopen(struct link_map *map, Lmid_t, uintptr_t *cookie) {
    *cookie = reinterpret_cast<uintptr_t>(map);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

extern "C" uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int, uintptr_t *reference_cookie,
                                   uintptr_t *definition_cookie, unsigned int *, const char *name) {
    const auto *from = reinterpret_cast<const link_map *>(*reference_cookie);
    const auto *to = reinterpret_cast<const link_map *>(*definition_cookie);
    const auto address = static_cast<std::uintptr_t>(symbol->st_value) +
                         static_cast<std::uintptr_t>(to ? to->l_addr : 0);
    record(name, from, to, address);
    return symbol->st_value;
}
