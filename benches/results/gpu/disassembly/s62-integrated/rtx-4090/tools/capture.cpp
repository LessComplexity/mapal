// Counter-free diagnostic prototype. No CUDA runtime/driver calls in callbacks.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
namespace {
constexpr size_t IMAGE_CAP=64u<<20,TOTAL_CAP=256u<<20,IMAGE_COUNT=128,RECORD_CAP=65536,BUFFER_CAP=8u<<20;
std::mutex mu;
FILE* log_=nullptr; int dir_=-1,mem_=-1;std::vector<char> expected_; size_t seq_=0,bytes_=0,images_=0,buffers_=0;
bool failed_=false,active_=false,finished_=false; unsigned starts_=0,stops_=0,selected_=0;
CUpti_SubscriberHandle sub_;
struct Frame {uint64_t invocation;uint32_t correlation;};
thread_local std::vector<Frame> runtime_;thread_local std::map<uint64_t,bool> payload_;
struct Invocation {unsigned domain,id;uint32_t correlation;const uint64_t* storage;};
uint64_t invocation_seq_=0;std::map<uint64_t,Invocation> invocations_;std::map<const uint64_t*,uint64_t> invocation_storage_;
void bad(const char* why){failed_=true;if(log_){fprintf(log_,"error\t%s\n",why);fflush(log_);}}
size_t ev(){if(++seq_>RECORD_CAP){bad("record-cap");return 0;}return seq_;}
std::string hex(const char* s){if(!s){bad("missing-string");return "";}size_t n=strnlen(s,16385);if(!n||n>16384){bad("string-cap");return "";}std::string o;o.reserve(n*2);const char* h="0123456789abcdef";for(size_t i=0;i<n;i++){unsigned char c=s[i];o+=h[c>>4];o+=h[c&15];}return o;}
uintptr_t ptr(const void* p){return reinterpret_cast<uintptr_t>(p);}
bool save(const char* name,const void* data,size_t n){int f=openat(dir_,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(f<0){bad("image-create");return false;}size_t off=0;while(off<n){ssize_t k=write(f,static_cast<const char*>(data)+off,n-off);if(k<=0){close(f);bad("image-write");return false;}off+=size_t(k);}if(close(f)){bad("image-close");return false;}return true;}
bool supported(CUpti_CallbackId id){return id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel||id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz||id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx||id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz;}
struct Launch {CUfunction f;unsigned gx,gy,gz,bx,by,bz,smem;CUstream stream;};
Launch launch(CUpti_CallbackId id,const void* raw){
 if(id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel){const auto* p=static_cast<const cuLaunchKernel_params*>(raw);return {p->f,p->gridDimX,p->gridDimY,p->gridDimZ,p->blockDimX,p->blockDimY,p->blockDimZ,p->sharedMemBytes,p->hStream};}
 if(id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz){const auto* p=static_cast<const cuLaunchKernel_ptsz_params*>(raw);return {p->f,p->gridDimX,p->gridDimY,p->gridDimZ,p->blockDimX,p->blockDimY,p->blockDimZ,p->sharedMemBytes,p->hStream};}
 const CUlaunchConfig* c=nullptr;CUfunction f=nullptr;
 if(id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx){const auto* p=static_cast<const cuLaunchKernelEx_params*>(raw);c=p->config;f=p->f;}
 else {const auto* p=static_cast<const cuLaunchKernelEx_ptsz_params*>(raw);c=p->config;f=p->f;}
 if(!c){bad("null-launch-config");return {};}if(c->numAttrs){bad("unsupported-launch-attributes");return {};}
 return {f,c->gridDimX,c->gridDimY,c->gridDimZ,c->blockDimX,c->blockDimY,c->blockDimZ,c->sharedMemBytes,c->hStream};
}
void CUPTIAPI callback(void*,CUpti_CallbackDomain dom,CUpti_CallbackId id,const void* raw){
 std::lock_guard<std::mutex> g(mu);if(failed_||finished_)return;
 if(dom==CUPTI_CB_DOMAIN_RESOURCE){const auto* r=static_cast<const CUpti_ResourceData*>(raw);
  if(id==CUPTI_CBID_RESOURCE_MODULE_LOADED){const auto* m=static_cast<const CUpti_ModuleResourceData*>(r->resourceDescriptor);
   if(!m||!m->pCubin||m->cubinSize<4||m->cubinSize>IMAGE_CAP||m->cubinSize>TOTAL_CAP-bytes_||images_>=IMAGE_COUNT){bad("image-cap-or-payload");return;}
   if(memcmp(m->pCubin,"\177ELF",4)){bad("image-not-ELF");return;}size_t e=ev();if(!e)return;
   char n[80];snprintf(n,sizeof n,"module-%zu.cubin",e);if(!save(n,m->pCubin,m->cubinSize))return;images_++;bytes_+=m->cubinSize;
   fprintf(log_,"image\t%zu\t%llu\t%u\t%zu\t%s\n",e,(unsigned long long)ptr(r->context),m->moduleId,m->cubinSize,n);
  }else if(id==CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING){const auto* m=static_cast<const CUpti_ModuleResourceData*>(r->resourceDescriptor);if(!m){bad("unload-no-payload");return;}fprintf(log_,"unload\t%zu\t%llu\t%u\n",ev(),(unsigned long long)ptr(r->context),m->moduleId);}
  else if(id==CUPTI_CBID_RESOURCE_CONTEXT_CREATED||id==CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING)fprintf(log_,"context\t%zu\t%u\t%llu\n",ev(),unsigned(id),(unsigned long long)ptr(r->context));
 }else if(dom==CUPTI_CB_DOMAIN_RUNTIME_API||dom==CUPTI_CB_DOMAIN_DRIVER_API){const auto* d=static_cast<const CUpti_CallbackData*>(raw);bool enter=d->callbackSite==CUPTI_API_ENTER;unsigned code=enter?0xffffffffu:(d->functionReturnValue?*static_cast<const unsigned*>(d->functionReturnValue):0xfffffffeu);std::string name=hex(d->functionName);
  // CUPTI explicitly provides invocation-specific ENTER/EXIT storage. Activity
  // correlation IDs may be shared by nested or repeated internal driver calls.
  if(!d->correlationData){bad("missing-invocation-storage");return;}
  uint64_t invocation=0;
  if(enter){if(invocation_storage_.count(d->correlationData)||invocations_.size()>=RECORD_CAP){bad("active-invocation-storage-collision");return;}invocation=++invocation_seq_;*d->correlationData=invocation;invocations_[invocation]={unsigned(dom),unsigned(id),d->correlationId,d->correlationData};invocation_storage_[d->correlationData]=invocation;}
  else {invocation=*d->correlationData;auto it=invocations_.find(invocation);if(it==invocations_.end()||it->second.domain!=unsigned(dom)||it->second.id!=unsigned(id)||it->second.correlation!=d->correlationId||it->second.storage!=d->correlationData){bad("unpaired-invocation-exit");return;}}

  fprintf(log_,"api\t%zu\t%u\t%u\t%u\t%u\t%llu\t%u\t%u\t%u\t%s\t%llu\n",ev(),unsigned(dom),unsigned(id),unsigned(enter),d->correlationId,(unsigned long long)ptr(d->context),d->contextUid,unsigned(active_),code,name.c_str(),(unsigned long long)invocation);
  if(dom==CUPTI_CB_DOMAIN_RUNTIME_API){
   if(enter)runtime_.push_back({invocation,d->correlationId});
   else {if(runtime_.empty()||runtime_.back().invocation!=invocation){bad("runtime-callback-stack");return;}runtime_.pop_back();}
   if(id==CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000&&!enter){if(code||active_||++starts_!=1){bad("invalid-region-start");return;}active_=true;fprintf(log_,"region-start\t%zu\t%u\t%u\n",ev(),d->contextUid,d->correlationId);}
   if(id==CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000){if(enter){if(!active_||++stops_!=1){bad("invalid-region-stop");return;}active_=false;fprintf(log_,"region-stop\t%zu\t%u\t%u\n",ev(),d->contextUid,d->correlationId);}else if(code)bad("region-stop-failed");}
  }else{
   if(supported(id)){if(!enter&&active_&&++selected_>512){bad("selected-kernel-cap");return;}Launch l=launch(id,d->functionParams);if(!enter&&code==0){uint32_t context_id=0,stream_id=0;uint8_t ptsz=id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz||id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz;CUptiResult cr=cuptiGetContextId(d->context,&context_id),sr=cuptiGetStreamIdEx(d->context,l.stream,ptsz,&stream_id);fprintf(log_,"stream-map\t%zu\t%u\t%llu\t%u\t%llu\t%u\t%u\t%u\t%u\t%llu\n",ev(),d->correlationId,(unsigned long long)ptr(d->context),context_id,(unsigned long long)ptr(l.stream),stream_id,unsigned(ptsz),unsigned(cr),unsigned(sr),(unsigned long long)invocation);if(cr!=CUPTI_SUCCESS||sr!=CUPTI_SUCCESS){bad("context-stream-metadata-query");return;}}std::string symbol=hex(d->symbolName);fprintf(log_,"launch\t%zu\t%u\t%u\t%u\t%u\t%llu\t%u\t%llu\t%llu\t%u,%u,%u\t%u,%u,%u\t%u\t%u\t%u\t%s\t%llu\t%llu\n",ev(),unsigned(id),unsigned(enter),d->correlationId,runtime_.empty()?0:runtime_.back().correlation,(unsigned long long)ptr(d->context),d->contextUid,(unsigned long long)ptr(l.f),(unsigned long long)ptr(l.stream),l.gx,l.gy,l.gz,l.bx,l.by,l.bz,l.smem,unsigned(active_),code,symbol.c_str(),(unsigned long long)invocation,(unsigned long long)(runtime_.empty()?0:runtime_.back().invocation));}
   else if(active_&&d->functionName&&strstr(d->functionName,"Launch"))bad("unsupported-selected-launch-API");
   if(id==CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData){const auto* p=static_cast<const cuModuleLoadData_params*>(d->functionParams);bool matched=false;if(enter){std::vector<char> candidate(expected_.size());ssize_t n=pread(mem_,candidate.data(),candidate.size(),(off_t)ptr(p->image));matched=n==ssize_t(candidate.size())&&candidate==expected_;payload_[invocation]=matched;}else{auto it=payload_.find(invocation);if(it==payload_.end()){bad("load-payload-enter-absent");return;}matched=it->second;payload_.erase(it);}fprintf(log_,"load\t%zu\t%u\t%u\t%llu\t%u\t%llu\t%u\t%u\t%llu\n",ev(),unsigned(enter),d->correlationId,(unsigned long long)ptr(d->context),d->contextUid,(unsigned long long)(!enter&&code==0&&p->module?ptr(*p->module):0),code,unsigned(matched),(unsigned long long)invocation);}
   if(!enter&&code==0&&id==CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction){const auto* p=static_cast<const cuModuleGetFunction_params*>(d->functionParams);std::string n=hex(p->name);fprintf(log_,"function-direct\t%zu\t%llu\t%u\t%llu\t%llu\t%s\t%llu\n",ev(),(unsigned long long)ptr(d->context),d->contextUid,(unsigned long long)ptr(p->hmod),(unsigned long long)(p->hfunc?ptr(*p->hfunc):0),n.c_str(),(unsigned long long)invocation);}
  }
  if(!enter){invocations_.erase(invocation);invocation_storage_.erase(d->correlationData);}
 }else if(dom==CUPTI_CB_DOMAIN_STATE)bad("CUPTI-state-notification");
 if(log_)fflush(log_);
}
void CUPTIAPI request_buffer(uint8_t** b,size_t* size,size_t* max){std::lock_guard<std::mutex> g(mu);*b=nullptr;*size=0;*max=0;if(failed_||buffers_+(1u<<20)>BUFFER_CAP){bad("activity-buffer-cap");return;}void* p=nullptr;if(posix_memalign(&p,8,1u<<20)){bad("activity-buffer-allocation");return;}*b=static_cast<uint8_t*>(p);*size=1u<<20;buffers_+=*size;}
void record(const CUpti_Activity* a){std::lock_guard<std::mutex> g(mu);if(failed_||finished_)return;
 if(a->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){const auto* k=reinterpret_cast<const CUpti_ActivityKernel9*>(a);std::string n=hex(k->name);fprintf(log_,"kernel\t%zu\t%u\t%u\t%u\t%lld\t%d,%d,%d\t%d,%d,%d\t%d\t%u\t%llu\t%u\t%s\n",ev(),k->correlationId,k->contextId,k->streamId,(long long)k->gridId,k->gridX,k->gridY,k->gridZ,k->blockX,k->blockY,k->blockZ,k->dynamicSharedMemory,unsigned(k->launchType),(unsigned long long)k->graphNodeId,k->graphId,n.c_str());}
 else if(a->kind==CUPTI_ACTIVITY_KIND_FUNCTION){const auto* f=reinterpret_cast<const CUpti_ActivityFunction*>(a);std::string n=hex(f->name);fprintf(log_,"function-activity\t%zu\t%u\t%u\t%u\t%u\t%s\n",ev(),f->id,f->contextId,f->moduleId,f->functionIndex,n.c_str());}
 else if(a->kind==CUPTI_ACTIVITY_KIND_DRIVER||a->kind==CUPTI_ACTIVITY_KIND_RUNTIME||a->kind==CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API){const auto* p=reinterpret_cast<const CUpti_ActivityAPI*>(a);fprintf(log_,"api-activity\t%zu\t%u\t%u\t%u\t%u\t%u\t%u\n",ev(),unsigned(a->kind),p->cbid,p->correlationId,p->returnValue,p->processId,p->threadId);}
 else if(a->kind==CUPTI_ACTIVITY_KIND_CONTEXT){const auto* c=reinterpret_cast<const CUpti_ActivityContext3*>(a);fprintf(log_,"context-activity\t%zu\t%u\t%u\n",ev(),c->contextId,c->deviceId);}
 else bad("unexpected-activity-kind");
}
void CUPTIAPI complete_buffer(CUcontext ctx,uint32_t stream,uint8_t* b,size_t size,size_t valid){CUpti_Activity* r=nullptr;CUptiResult result=CUPTI_SUCCESS;while((result=cuptiActivityGetNextRecord(b,valid,&r))==CUPTI_SUCCESS)record(r);size_t dropped=0;CUptiResult drop=cuptiActivityGetNumDroppedRecords(ctx,stream,&dropped);{std::lock_guard<std::mutex> g(mu);if(result!=CUPTI_ERROR_MAX_LIMIT_REACHED||drop!=CUPTI_SUCCESS||dropped)bad("activity-parse-or-drop");if(log_)fprintf(log_,"buffer\t%zu\t%zu\t%zu\t%zu\t%u\t%u\n",ev(),size,valid,dropped,unsigned(result),unsigned(drop));if(size>buffers_)bad("activity-buffer-accounting");else buffers_-=size;}free(b);}
void finish(){CUptiResult rc=cuptiActivityFlushAll(0);size_t dropped=0;CUptiResult dr=cuptiActivityGetNumDroppedRecords(nullptr,0,&dropped);std::lock_guard<std::mutex> g(mu);if(finished_)return;if(rc!=CUPTI_SUCCESS||dr!=CUPTI_SUCCESS||dropped||buffers_||!invocations_.empty()||active_||starts_!=1||stops_!=1)bad("incomplete-final-flush-or-region");if(log_){fprintf(log_,"final-drop\t%zu\t%zu\t%u\n",ev(),dropped,unsigned(dr));fprintf(log_,"finish\t%zu\t%u\t%u\t%u\t%u\t%zu\t%zu\t%zu\n",ev(),unsigned(rc),unsigned(failed_),starts_,stops_,images_,bytes_,buffers_);fflush(log_);}finished_=true;}
}
extern "C" int InitializeInjection(){const char* out=getenv("MAPAL_CAPTURE_OUT"),*ptx=getenv("MAPAL_CAPTURE_PTX");if(!out||!ptx||mkdir(out,0700))return 0;dir_=open(out,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);if(dir_<0)return 0;int fd=openat(dir_,"events.tsv",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)return 0;log_=fdopen(fd,"w");if(!log_)return 0;fprintf(log_,"schema\tlibrary-image-callback-v2\n");int input=open(ptx,O_RDONLY|O_NOFOLLOW);struct stat st{};if(input<0||fstat(input,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||st.st_size>(16<<20)){bad("PTX-input");return 0;}expected_.resize(size_t(st.st_size)+1);size_t pos=0;while(pos<size_t(st.st_size)){ssize_t n=read(input,expected_.data()+pos,size_t(st.st_size)-pos);if(n<=0){bad("PTX-read");close(input);return 0;}pos+=size_t(n);}close(input);if(memchr(expected_.data(),0,pos)||!save("input.ptx",expected_.data(),pos)){bad("PTX-format-or-save");return 0;}mem_=open("/proc/self/mem",O_RDONLY);
 if(cuptiSubscribe(&sub_,callback,nullptr)!=CUPTI_SUCCESS){bad("subscribe");return 0;}
 for(auto d:{CUPTI_CB_DOMAIN_DRIVER_API,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CB_DOMAIN_STATE})if(cuptiEnableDomain(1,sub_,d)!=CUPTI_SUCCESS){bad("enable-domain");return 0;}
 if(cuptiActivityRegisterCallbacks(request_buffer,complete_buffer)!=CUPTI_SUCCESS){bad("register-activity");return 0;}
 for(auto k:{CUPTI_ACTIVITY_KIND_DRIVER,CUPTI_ACTIVITY_KIND_RUNTIME,CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API,CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_FUNCTION,CUPTI_ACTIVITY_KIND_CONTEXT}){CUptiResult r=cuptiActivityEnable(k);fprintf(log_,"enable\t%u\t%u\n",unsigned(k),unsigned(r));if(r!=CUPTI_SUCCESS&&k!=CUPTI_ACTIVITY_KIND_FUNCTION){bad("enable-activity");return 0;}}
 uint32_t v=0;CUptiResult rc=cuptiGetVersion(&v);if(rc!=CUPTI_SUCCESS){bad("get-version");return 0;}fprintf(log_,"ready\t%u\n",v);fflush(log_);if(atexit(finish)){bad("atexit");return 0;}return 1;}
