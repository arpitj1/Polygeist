#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
extern "C" void *mgpuMemAlloc(uint64_t, void *, uint8_t);
extern "C" void mgpuMemFree(void *, void *);
#define CU(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));abort();}}while(0)
static void check(bool okay, const char *message) {if(!okay){fprintf(stderr,"FAIL %s\n",message);abort();}}
static void pause_stream(void *) {usleep(100000);}
int main() {
 check(mgpuMemAlloc(0,nullptr,0)==nullptr,"zero allocation");mgpuMemFree(nullptr,nullptr);
 void *first=mgpuMemAlloc(4096,nullptr,0),*second=mgpuMemAlloc(4096,nullptr,0);
 check(first!=second,"exclusive simultaneous leases");
 CU(cudaMemset(first,0x3c,4096));CU(cudaMemset(second,0x5a,4096));
 mgpuMemFree(first,nullptr);
 void *reused=mgpuMemAlloc(3072,nullptr,0);
 // Cache-on runs require best-fit reuse. Cache-off results check semantics only.
 bool enabled=getenv("POLYGEIST_GPU_WORKSPACE_CACHE") && atoi(getenv("POLYGEIST_GPU_WORKSPACE_CACHE"));
 if(enabled)check(reused==first,"best-fit allocation reuse");
 unsigned char value=0;CU(cudaMemcpy(&value,second,1,cudaMemcpyDeviceToHost));check(value==0x5a,"active lease unchanged");
 cudaStream_t stream;cudaEvent_t event;CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CU(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
 CU(cudaLaunchHostFunc(stream,pause_stream,nullptr));CU(cudaMemsetAsync(reused,0x71,3072,stream));CU(cudaEventRecord(event,stream));
 check(cudaEventQuery(event)==cudaErrorNotReady,"producer really pending");
 mgpuMemFree(reused,stream);check(cudaEventQuery(event)==cudaSuccess,"release completes pending producers");
 mgpuMemFree(second,nullptr);CU(cudaEventDestroy(event));CU(cudaStreamDestroy(stream));
 // Two host threads must never receive the same active workspace.
 std::atomic<int> ready{0};void *leased[2]={nullptr,nullptr};
 auto worker=[&](int id){CU(cudaSetDevice(0));leased[id]=mgpuMemAlloc(4096,nullptr,0);ready.fetch_add(1);while(ready.load()!=2)std::this_thread::yield();check(leased[0]!=leased[1],"thread lease isolation");CU(cudaMemset(leased[id],id+1,4096));mgpuMemFree(leased[id],nullptr);};
 std::thread a(worker,0),b(worker,1);a.join();b.join();
 // Requests larger than the configured 16 KiB cache must remain ordinary CUDA
 // allocations and must not change retained byte accounting.
 void *oversized=mgpuMemAlloc(65536,nullptr,0);CU(cudaMemset(oversized,0xa4,65536));mgpuMemFree(oversized,nullptr);
 void *host=mgpuMemAlloc(4096,nullptr,1);cudaPointerAttributes attrs;CU(cudaPointerGetAttributes(&attrs,host));check(attrs.type==cudaMemoryTypeHost,"mapped host allocation");static_cast<unsigned char*>(host)[0]=0x6d;mgpuMemFree(host,nullptr);
 // Reset destroys old cached storage. A fresh context must never lease it;
 // virtual addresses are allowed to repeat, so context ID is the real guard.
 CU(cudaDeviceReset());void *after_reset=mgpuMemAlloc(4096,nullptr,0);CU(cudaMemset(after_reset,0x17,4096));CU(cudaMemcpy(&value,after_reset,1,cudaMemcpyDeviceToHost));check(value==0x17,"fresh context allocation");mgpuMemFree(after_reset,nullptr);
 void *after_reinit=mgpuMemAlloc(4096,nullptr,0);CU(cudaMemset(after_reinit,0x29,4096));CU(cudaMemcpy(&value,after_reinit,1,cudaMemcpyDeviceToHost));check(value==0x29,"reinitialized context allocation");mgpuMemFree(after_reinit,nullptr);
 puts("PASS workspace leases, reuse, completion, thread isolation, capacity fallback, mapped-host free, zero allocation and context reset");
}
