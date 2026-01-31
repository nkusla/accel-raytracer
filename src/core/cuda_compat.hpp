#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#define __host__
#define __global__
#define __forceinline__
#endif