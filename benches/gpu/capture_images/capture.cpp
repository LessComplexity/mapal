// Observation only. Linux, CUDA/CUPTI 12.8.90. No CUDA calls or synchronization.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <mutex>
#include <map>
#include <string>
#include <vector>
namespace {
constexpr size_t CAP=16u<<20, EVENTS=4096;
std::mutex mu;
FILE *logfile=nullptr;
int dirfd_=-1, memfd_=-1;
CUpti_SubscriberHandle subscriber;
std::vector<char> expected;
size_t seq=0, captured_bytes=0; bool failed=false;
struct Module {unsigned id, context; size_t load;};
struct Function {Module module; std::string name;};
std::map<std::pair<CUcontext,CUmodule>,Module> modules;
std::map<std::pair<CUcontext,CUfunction>,Function> functions;
struct Pending {bool matched=false; unsigned count=0,id=0; size_t load=0; CUcontext context=nullptr; unsigned context_uid=0,correlation=0; CUmodule *module_out=nullptr; const void *image=nullptr;} ;
thread_local Pending pending;
void error(const char *s) {failed=true; if(logfile)fprintf(logfile,"error\t%s\n",s);}
size_t event() {if(++seq>EVENTS){error("event-cap");return 0;}return seq;}
bool writefile(const char *name,const void *p,size_t n) {
 int f=openat(dirfd_,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
 if(f<0){error("create-file");return false;}
 const char *b=(const char*)p; size_t i=0;
 while(i<n){ssize_t k=write(f,b+i,n-i);if(k<=0){close(f);error("write-file");return false;}i+=size_t(k);}
 if(close(f)){error("close-file");return false;}return true;
}
void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void *raw) {
 std::lock_guard<std::mutex> guard(mu);
 if(failed)return;
 if(domain==CUPTI_CB_DOMAIN_RESOURCE && id==CUPTI_CBID_RESOURCE_MODULE_LOADED){
  if(!pending.matched)return; // Never persist unrelated/library modules.
  const auto *r=(const CUpti_ResourceData*)raw;
  if(!r->context||r->context!=pending.context){error("resource-context-mismatch");return;}
  const auto *m=(const CUpti_ModuleResourceData*)r->resourceDescriptor;
  if(!m||!m->pCubin||m->cubinSize<4||m->cubinSize>CAP-captured_bytes){error("invalid-cubin");return;}
  if(++pending.count!=1){error("ambiguous-module-callback");return;}
  if(memcmp(m->pCubin,"\177ELF",4)){error("not-elf");return;}
  captured_bytes+=m->cubinSize; pending.id=m->moduleId; auto e=event(); if(!e)return;
  char name[80];snprintf(name,sizeof name,"module-%zu.cubin",pending.load);
  if(writefile(name,m->pCubin,m->cubinSize))
   fprintf(logfile,"image\t%zu\t%zu\t%p\t%u\t%zu\t%s\n",e,pending.load,(void*)r->context,m->moduleId,m->cubinSize,name);
  return;
 }
 if(domain!=CUPTI_CB_DOMAIN_DRIVER_API)return;
 const auto *d=(const CUpti_CallbackData*)raw;
 if(id==CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData){
  const auto *p=(const cuModuleLoadData_params*)d->functionParams;
  if(d->callbackSite==CUPTI_API_ENTER){
   if(pending.matched){error("nested-load");return;}
   // Safe read of only the authorized PTX-sized region; unknown images are not retained.
   std::vector<char> candidate(expected.size());
   auto n=pread(memfd_,candidate.data(),candidate.size(),(off_t)(uintptr_t)p->image);
   if(n!=(ssize_t)expected.size()||candidate!=expected)return;
   if(!d->context){error("missing-load-context");return;}
   pending={true,0,0,event(),d->context,d->contextUid,d->correlationId,p->module,p->image};
   fprintf(logfile,"load-enter\t%zu\t%p\t%u\t%u\n",pending.load,(void*)d->context,d->contextUid,d->correlationId);
  }else if(pending.matched){
   if(d->callbackSite!=CUPTI_API_EXIT||d->context!=pending.context||d->contextUid!=pending.context_uid||d->correlationId!=pending.correlation||p->module!=pending.module_out||p->image!=pending.image){error("load-exit-identity-mismatch");return;}
   if(!d->functionReturnValue||*(const CUresult*)d->functionReturnValue!=CUDA_SUCCESS||pending.count!=1){error("load-failed-or-no-image");return;}
   Module m{pending.id,d->contextUid,pending.load};
   if(!modules.emplace(std::make_pair(d->context,*p->module),m).second){error("module-handle-reused");return;}
   auto e=event(); if(!e)return;
   fprintf(logfile,"load-exit\t%zu\t%zu\t%p\t%u\t%u\t%u\n",e,m.load,(void*)d->context,m.context,m.id,d->correlationId);
   pending={};
  }
 } else if(d->callbackSite==CUPTI_API_EXIT && id==CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction){
  const auto *p=(const cuModuleGetFunction_params*)d->functionParams;
  auto m=modules.find({d->context,p->hmod});if(m==modules.end())return;
  if(!d->functionReturnValue||*(const CUresult*)d->functionReturnValue!=CUDA_SUCCESS){error("get-function-failed");return;}
  size_t n=strnlen(p->name,1025);if(!n||n>1024){error("function-name-cap");return;}
  std::string name(p->name,n);if(name.find_first_of("\t\r\n")!=std::string::npos){error("function-name-format");return;}
  auto key=std::make_pair(d->context,*p->hfunc);auto f=functions.find(key);
  if(f!=functions.end()&&(f->second.name!=name||f->second.module.load!=m->second.load)){error("function-handle-reused");return;}
  functions[key]={m->second,name};
  fprintf(logfile,"function\t%zu\t%zu\t%u\t%u\t%p\t%s\n",event(),m->second.load,m->second.context,m->second.id,(void*)*p->hfunc,name.c_str());
 } else if(d->callbackSite==CUPTI_API_EXIT && id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel){
  const auto *p=(const cuLaunchKernel_params*)d->functionParams;
  auto f=functions.find({d->context,p->f});if(f==functions.end())return;
  if(!d->functionReturnValue||*(const CUresult*)d->functionReturnValue!=CUDA_SUCCESS){error("launch-failed");return;}
  auto m=f->second.module;
  fprintf(logfile,"launch\t%zu\t%zu\t%u\t%u\t%p\t%u\t%u,%u,%u\t%u,%u,%u\t%u\n",event(),m.load,m.context,m.id,(void*)p->f,d->correlationId,p->gridDimX,p->gridDimY,p->gridDimZ,p->blockDimX,p->blockDimY,p->blockDimZ,p->sharedMemBytes);
 }
 fflush(logfile);
}
}
extern "C" int InitializeInjection() {
 const char *out=getenv("MAPAL_CAPTURE_OUT"),*ptx=getenv("MAPAL_CAPTURE_PTX");
 if(!out||!ptx||mkdir(out,0700)){fprintf(stderr,"capture: missing config or output exists\n");return 0;}
 dirfd_=open(out,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);memfd_=open("/proc/self/mem",O_RDONLY);
 if(dirfd_<0||memfd_<0)return 0;
 int f=open(ptx,O_RDONLY|O_NOFOLLOW);struct stat st{};
 if(f<0||fstat(f,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||st.st_size>(off_t)CAP)return 0;
 expected.resize(st.st_size+1);size_t done=0;
 while(done<(size_t)st.st_size){ssize_t n=read(f,expected.data()+done,st.st_size-done);if(n<=0)return 0;done+=n;}close(f);
 if(memchr(expected.data(),0,st.st_size))return 0;
 int lf=openat(dirfd_,"events.tsv",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(lf<0)return 0;
 logfile=fdopen(lf,"w");if(!logfile)return 0;
 if(!writefile("input.ptx",expected.data(),expected.size()-1))return 0;
 fprintf(logfile,"schema\tmapal-cupti-driver-v2\n");
 if(cuptiSubscribe(&subscriber,callback,nullptr)!=CUPTI_SUCCESS){error("subscribe");fflush(logfile);return 0;}
 const CUpti_CallbackId ids[]={CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel};
 for(auto id:ids)if(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,id)!=CUPTI_SUCCESS){error("enable-driver");fflush(logfile);return 0;}
 if(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED)!=CUPTI_SUCCESS){error("enable-resource");fflush(logfile);return 0;}
 fprintf(logfile,"ready\n");fflush(logfile);return 1;
}
