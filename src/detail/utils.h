#pragma once

#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cuda_cuda_h__
inline bool check(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(e, &err);
        std::cerr << "CUDA error " << e << " at line " << iLine << " in file " << szFile
                  << ": " << err << std::endl;
        return false;
    }
    return true;
}
#endif

#ifdef __CUDA_RUNTIME_H__
inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA runtime error " << e << " at line " << iLine
                  << " in file " << szFile
                  << ": " << cudaGetErrorString(e)
                  << std::endl;
        return false;
    }
    return true;
}
#endif

#define cucall(call) check(call, __LINE__, __FILE__)
