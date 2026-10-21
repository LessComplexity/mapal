// Real pinned CUPTI types, stubbed CUPTI functions; never a CUDA execution test.
#include "capture.cpp"
#include <cassert>
extern "C" CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle*,CUpti_CallbackFunc,void*) {return CUPTI_SUCCESS;}
extern "C" CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t,CUpti_SubscriberHandle,CUpti_CallbackDomain,CUpti_CallbackId) {return CUPTI_SUCCESS;}
int main(int argc,char **argv){
 assert(argc==4);const std::string mode=argv[2];
 assert(mkdir(argv[1],0700)==0);dirfd_=open(argv[1],O_RDONLY|O_DIRECTORY);
 int fd=openat(dirfd_,"events.tsv",O_RDWR|O_CREAT|O_EXCL,0600);logfile=fdopen(fd,"w");
 fprintf(logfile,"schema\tmapal-cupti-driver-v2\nready\n");
 expected={'/','/','x',0}; assert(writefile("input.ptx",expected.data(),3));
 memfd_=openat(dirfd_,"memory",O_RDWR|O_CREAT|O_EXCL,0600);assert(write(memfd_,expected.data(),4)==4);
 CUmodule mod=(CUmodule)0x11;CUfunction fn=(CUfunction)0x22;CUresult success=CUDA_SUCCESS;
 CUpti_CallbackData d{};d.context=(CUcontext)0x33;d.contextUid=4;d.correlationId=7;
 cuModuleLoadData_params lp{&mod,(const void*)0};d.functionParams=&lp;d.callbackSite=CUPTI_API_ENTER;
 callback(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,&d);
 int cf=open(argv[3],O_RDONLY);assert(cf>=0); std::vector<char> cubin(4096);ssize_t cn=read(cf,cubin.data(),cubin.size());assert(cn>0);cubin.resize(cn);close(cf);
 CUpti_ModuleResourceData mr{};mr.moduleId=9;mr.pCubin=cubin.data();mr.cubinSize=cubin.size();
 CUpti_ResourceData r{};r.context=d.context;r.resourceDescriptor=&mr;
 if(mode=="resource-wrong-context")r.context=(CUcontext)0x44;
 if(mode=="empty")mr.cubinSize=0;
 if(mode=="oversize")mr.cubinSize=CAP+1;
 if(mode=="cap")seq=EVENTS;
 if(mode=="corrupt")cubin[0]='X';
 if(mode!="missing")callback(nullptr,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED,&r);
 if(mode=="duplicate")callback(nullptr,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED,&r);
 // Mutate callback storage immediately: saved image must retain original bytes.
 cubin[4]='Z';
 d.callbackSite=CUPTI_API_EXIT;d.functionReturnValue=&success;
 if(mode=="exit-wrong-context")d.context=(CUcontext)0x44;
 if(mode=="exit-wrong-uid")d.contextUid=5;
 if(mode=="exit-wrong-correlation")d.correlationId=8;
 if(mode=="exit-wrong-output")lp.module=nullptr;
 if(mode=="exit-wrong-image")lp.image=(const void*)0x44;
 callback(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,&d);
 cuModuleGetFunction_params fp{&fn,mod,"kernel"};d.functionParams=&fp;
 if(mode!="unlinked")callback(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,&d);
 cuLaunchKernel_params kp{};kp.f=fn;kp.gridDimX=kp.gridDimY=kp.gridDimZ=kp.blockDimX=kp.blockDimY=kp.blockDimZ=1;d.functionParams=&kp;
 if(mode=="launch-failed")success=CUDA_ERROR_INVALID_VALUE;
 if(mode!="no-launch")callback(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,&d);
 fflush(logfile);fclose(logfile);
 if(mode=="happy")assert(!failed);else if(mode!="unlinked"&&mode!="no-launch")assert(failed);
 return 0;
}
