#pragma once

// CUDA support: include cuda.h first to define CUDA_VERSION for GLM
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

// Define GLM settings for CUDA compatibility
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#ifndef GLM_FORCE_PURE
#define GLM_FORCE_PURE
#endif

#else
// Non-CUDA builds: provide compatible macros
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#endif

#include <glm/glm.hpp>