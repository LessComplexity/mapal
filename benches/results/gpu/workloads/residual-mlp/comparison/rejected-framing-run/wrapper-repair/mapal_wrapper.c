#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
extern int mapal_request(void);
extern double mapal_time_ms(void);
static CUevent begin,end;static int phase=0,rep=0;
static void ck(CUresult e){if(e!=CUDA_SUCCESS){fprintf(stderr,"CUDA event failure %d\n",(int)e);exit(105);}}
double mapal_bench_time_ms(void){
 if(!phase){double wall=mapal_time_ms();if(!begin){ck(cuEventCreate(&begin,CU_EVENT_DEFAULT));ck(cuEventCreate(&end,CU_EVENT_DEFAULT));}ck(cuEventRecord(begin,0));phase=1;return wall;}
 ck(cuEventRecord(end,0));ck(cuEventSynchronize(end));double wall=mapal_time_ms();float ms;ck(cuEventElapsedTime(&ms,begin,end));fprintf(stderr,"device_event_ms rep=%d ms=%.9g\n",rep,(double)ms);phase=0;return wall;
}
int main(int argc,char**argv){int count=argc==2?1:35;for(rep=0;rep<count;++rep){printf("sample\n%d\n",rep);fflush(stdout);if(mapal_request()!=0||phase)return 1;}printf("0\n");ck(cuEventDestroy(begin));ck(cuEventDestroy(end));return 0;}
