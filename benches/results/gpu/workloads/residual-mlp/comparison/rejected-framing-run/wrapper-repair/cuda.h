#ifndef TEST_CUDA_H
#define TEST_CUDA_H
typedef int CUresult;
typedef unsigned long CUevent;
#define CUDA_SUCCESS 0
#define CU_EVENT_DEFAULT 0
CUresult cuEventCreate(CUevent*, unsigned int);
CUresult cuEventRecord(CUevent, void*);
CUresult cuEventSynchronize(CUevent);
CUresult cuEventElapsedTime(float*,CUevent,CUevent);
CUresult cuEventDestroy(CUevent);
#endif
