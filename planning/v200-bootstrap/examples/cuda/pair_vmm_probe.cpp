// Qualification example only: not an interposer or a V200 implementation.
// Run through the existing CUDA controller with two explicit physical UUIDs.
// Defaults allocate a small range; no clocks, reset, peer-enable, or host policy changes.
#include <cuda.h>
#include "v200/planning.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(CUresult r, const char* operation) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr; const char* message = nullptr;
    cuGetErrorName(r, &name); cuGetErrorString(r, &message);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA error") + " " + (message ? message : ""));
}
#define CK(x) ck((x), #x)
static unsigned cleanup_failures = 0;
static void cleanup(CUresult r, const char* what) noexcept {
    if (r != CUDA_SUCCESS) { ++cleanup_failures; std::fprintf(stderr, "cleanup failure: %s (%d)\n", what, static_cast<int>(r)); }
}
struct ContextScope {
    explicit ContextScope(CUcontext c) { CK(cuCtxPushCurrent(c)); }
    ~ContextScope() { CUcontext ignored{}; cleanup(cuCtxPopCurrent(&ignored), "context pop"); }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
};
struct Primaries {
    std::array<CUdevice,2> d{}; std::array<CUcontext,2> c{}; unsigned held=0;
    explicit Primaries(std::array<CUdevice,2> devices):d(devices) {
        try { for (; held<2; ++held) CK(cuDevicePrimaryCtxRetain(&c[held], d[held])); }
        catch (...) { release(); throw; }
    }
    void release() noexcept { while (held) { --held; cleanup(cuDevicePrimaryCtxRelease(d[held]), "primary release"); } }
    ~Primaries(){ release(); }
    Primaries(const Primaries&) = delete;
    Primaries& operator=(const Primaries&) = delete;
};
static CUmemAllocationProp property(CUdevice d) {
    CUmemAllocationProp p{}; p.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type=CU_MEM_LOCATION_TYPE_DEVICE; p.location.id=d; return p;
}
struct Arena {
    CUdeviceptr base=0; std::size_t extent=0,total=0; unsigned mapped=0,created=0;
    std::array<CUmemGenericAllocationHandle,2> handle{};
    Arena(std::array<CUdevice,2> d, std::size_t requested, std::uint64_t allocation_cap) {
        try {
            auto pa=property(d[0]), pb=property(d[1]); std::size_t ga{},gb{};
            CK(cuMemGetAllocationGranularity(&ga,&pa,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            CK(cuMemGetAllocationGranularity(&gb,&pb,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            extent=v200::round_up(requested,v200::common_granularity(ga,gb));
            if (extent>64ULL*1024*1024) throw std::runtime_error("rounded probe extent exceeds 64 MiB cap per owner");
            if (v200::add(extent,sizeof(unsigned)) > allocation_cap) throw std::runtime_error("probe exceeds bundle application allocation cap");
            total=v200::mul(extent,2);
            CK(cuMemAddressReserve(&base,total,0,0,0));
            for (unsigned i=0;i<2;++i) {
                auto p=property(d[i]); CK(cuMemCreate(&handle[i],extent,&p,0)); ++created;
                CK(cuMemMap(base+i*extent,extent,0,handle[i],0)); ++mapped;
            }
            CUmemAccessDesc access[2]{};
            for (unsigned i=0;i<2;++i) { access[i].location.type=CU_MEM_LOCATION_TYPE_DEVICE;
                access[i].location.id=d[i]; access[i].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
            CK(cuMemSetAccess(base,total,access,2));
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        while(mapped) { --mapped; cleanup(cuMemUnmap(base+mapped*extent,extent),"extent unmap"); }
        while(created) { --created; cleanup(cuMemRelease(handle[created]),"backing release"); }
        if(base) { cleanup(cuMemAddressFree(base,total),"VA release"); base=0; }
    }
    ~Arena(){ release(); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
};
// These entry points use explicit global element indices. They do not rewrite
// blockIdx/gridDim and do not claim a generic split-kernel transformation.
static const char* ptx=R"PTX(
.version 7.0
.target sm_70
.address_size 64
.visible .entry fill(.param .u64 ptr, .param .u64 first, .param .u64 count) {
 .reg .pred p; .reg .b32 r<6>; .reg .b64 d<9>;
 ld.param.u64 d0,[ptr]; ld.param.u64 d1,[first]; ld.param.u64 d2,[count];
 mov.u32 r0,%ctaid.x; mov.u32 r1,%ntid.x; mov.u32 r2,%tid.x;
 mad.lo.u32 r3,r0,r1,r2; cvt.u64.u32 d3,r3;
 setp.ge.u64 p,d3,d2; @p bra Fdone;
 add.u64 d4,d3,d1; mul.lo.u64 d5,d4,4; add.u64 d6,d0,d5;
 cvt.u32.u64 r4,d4; xor.b32 r5,r4,0x9e3779b9; st.global.u32 [d6],r5;
Fdone: ret;
}
.visible .entry verify(.param .u64 ptr, .param .u64 count, .param .u64 errors) {
 .reg .pred p; .reg .b32 r<9>; .reg .b64 d<9>;
 ld.param.u64 d0,[ptr]; ld.param.u64 d1,[count]; ld.param.u64 d2,[errors];
 mov.u32 r0,%ctaid.x; mov.u32 r1,%ntid.x; mov.u32 r2,%tid.x;
 mad.lo.u32 r3,r0,r1,r2; cvt.u64.u32 d3,r3;
 setp.ge.u64 p,d3,d1; @p bra Vdone;
 mul.lo.u64 d4,d3,4; add.u64 d5,d0,d4; ld.global.u32 r4,[d5];
 xor.b32 r5,r3,0x9e3779b9; setp.eq.u32 p,r4,r5; @p bra Vdone;
 atom.global.add.u32 r6,[d2],1;
Vdone: ret;
}
)PTX";
struct Worker {
    CUcontext c{}; CUstream stream{}; CUevent done{}; CUmodule module{}; CUdeviceptr errors{};
    CUfunction fill{},verify{};
    explicit Worker(CUcontext context):c(context) {
        try { ContextScope scope(c); CK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
            CK(cuEventCreate(&done,CU_EVENT_DISABLE_TIMING)); CK(cuModuleLoadData(&module,ptx));
            CK(cuModuleGetFunction(&fill,module,"fill")); CK(cuModuleGetFunction(&verify,module,"verify"));
            CK(cuMemAlloc(&errors,sizeof(unsigned))); }
        catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (!c) return;
        if (cuCtxPushCurrent(c)!=CUDA_SUCCESS) { ++cleanup_failures; std::fprintf(stderr,"cleanup context unavailable\n"); c=nullptr; return; }
        if(stream) cleanup(cuStreamSynchronize(stream),"owned stream drain");
        if(errors) { cleanup(cuMemFree(errors),"error buffer"); errors=0; }
        if(module) { cleanup(cuModuleUnload(module),"module unload"); module=nullptr; }
        if(done) { cleanup(cuEventDestroy(done),"event destroy"); done=nullptr; }
        if(stream) { cleanup(cuStreamDestroy(stream),"stream destroy"); stream=nullptr; }
        CUcontext ignored{}; cleanup(cuCtxPopCurrent(&ignored),"worker pop"); c=nullptr;
    }
    ~Worker(){ release(); }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
};
static std::string uuid(CUdevice d) {
    CUuuid u{}; CK(cuDeviceGetUuid(&u,d));
    std::ostringstream o; o<<"GPU-"<<std::hex<<std::setfill('0');
    for(unsigned i=0;i<16;++i){if(i==4||i==6||i==8||i==10)o<<'-';o<<std::setw(2)<<static_cast<unsigned>(static_cast<unsigned char>(u.bytes[i]));}
    return o.str();
}
static void launch_fill(Worker& w,CUdeviceptr ptr,std::uint64_t first,std::uint64_t count) {
    ContextScope scope(w.c); void* args[]{&ptr,&first,&count};
    CK(cuLaunchKernel(w.fill,static_cast<unsigned>((count+255)/256),1,1,256,1,1,0,w.stream,args,nullptr));
}
static unsigned verify_all(Worker& w,CUdeviceptr ptr,std::uint64_t count) {
    ContextScope scope(w.c); CK(cuMemsetD32Async(w.errors,0,1,w.stream)); void* args[]{&ptr,&count,&w.errors};
    CK(cuLaunchKernel(w.verify,static_cast<unsigned>((count+255)/256),1,1,256,1,1,0,w.stream,args,nullptr));
    CK(cuStreamSynchronize(w.stream)); unsigned errors{}; CK(cuMemcpyDtoH(&errors,w.errors,sizeof(errors))); return errors;
}
int main(int argc,char** argv) {
    std::string expect_a,expect_b; bool execute=false; std::size_t bytes=65536;
    try {
        for(int i=1;i<argc;++i){ const std::string a=argv[i];
            if(a=="--execute") execute=true;
            else if((a=="--expect-a"||a=="--expect-b"||a=="--bytes-per-owner")&&i+1<argc){
                std::string value=argv[++i];
                if(a=="--expect-a")expect_a=value; else if(a=="--expect-b")expect_b=value;
                else { if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("invalid byte count");
                    bytes=static_cast<std::size_t>(std::stoull(value)); }
            } else throw std::runtime_error("unknown or incomplete argument");
        }
        if(!execute||expect_a.empty()||expect_b.empty()||expect_a==expect_b||!bytes||bytes>64ULL*1024*1024)
            throw std::runtime_error("require --execute --expect-a GPU-UUID --expect-b GPU-UUID; bytes per owner 1..67108864");
        if(!std::getenv("TODO_GPU_LEASE_RECEIPT")) throw std::runtime_error("controller lease required; use the admitted bundle, not direct execution");
        // The bounded runner validates receipt ownership/ancestry and UUID set;
        // the environment's existence alone is not an authorization mechanism.
        const char* cap_text=std::getenv("V200_BUNDLE_VRAM_CAP_MIB_PER_GPU");
        if(!cap_text)throw std::runtime_error("bundle allocation cap missing");
        const auto cap=v200::mul(std::stoull(cap_text),1024ULL*1024);
        CK(cuInit(0)); int count{};CK(cuDeviceGetCount(&count));
        if(count!=2)throw std::runtime_error("exactly two controller-visible GPUs required");
        std::array<CUdevice,2>d{};CK(cuDeviceGet(&d[0],0));CK(cuDeviceGet(&d[1],1));
        std::array<std::string,2>u{uuid(d[0]),uuid(d[1])};
        if(u[0]!=expect_a||u[1]!=expect_b)throw std::runtime_error("visible order/UUID mismatch");
        for(auto dev:d){int major{},minor{},vmm{};CK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));
            CK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));
            CK(cuDeviceGetAttribute(&vmm,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
            if(major!=7||minor!=0||!vmm)throw std::runtime_error("sm_70 and VMM required");}
        for(unsigned i=0;i<2;++i){int peer{};CK(cuDeviceCanAccessPeer(&peer,d[i],d[1-i]));if(!peer)throw std::runtime_error("bidirectional peer capability required");}
        std::size_t measured_extent=0; std::uint64_t measured_count=0;
        {
        Primaries primaries(d);
        ContextScope anchor(primaries.c[0]);
        Arena arena(d,bytes,cap);
        Worker a(primaries.c[0]), b(primaries.c[1]);
        const auto n=static_cast<std::uint64_t>(arena.extent/sizeof(unsigned));
        launch_fill(a,arena.base,0,n);
        launch_fill(b,arena.base,n,n);
        {ContextScope scope(b.c);CK(cuEventRecord(b.done,b.stream));}
        {ContextScope scope(a.c);CK(cuStreamWaitEvent(a.stream,b.done,0));CK(cuEventRecord(a.done,a.stream));}
        {ContextScope scope(b.c);CK(cuStreamWaitEvent(b.stream,a.done,0));}
        // After fan-in, each GPU reads the whole canonical range through its
        // own SMs, including peer-owned memory. Its error atomic stays local.
        auto errors_a=verify_all(a,arena.base,2*n), errors_b=verify_all(b,arena.base,2*n);
        if(errors_a||errors_b)throw std::runtime_error("pattern mismatch");
        measured_extent=arena.extent; measured_count=2*n;
        } // all owned CUDA resources are released before reporting success
        if(cleanup_failures)throw std::runtime_error("owned-resource cleanup failed");
        std::cout<<"{\"status\":\"GPU_PASS\",\"qualification_only\":true,\"requested_bytes_per_owner\":"<<bytes
          <<",\"mapped_bytes_per_owner\":"<<measured_extent<<",\"checked_u32_per_device\":"<<measured_count
          <<",\"errors\":0,\"arbitrary_kernel_transparency\":false,\"nvlink_route_proven\":false}\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<"pair probe: "<<e.what()<<'\n';return 1;}
}
