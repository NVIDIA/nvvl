#pragma once

#include <cuda.h>

namespace NVVL {
namespace detail {

class CUContext {
  public:
    CUContext();
    CUContext(CUdevice device, unsigned int flags = 0);
    CUContext(CUcontext);
    ~CUContext();

    // no copying
    CUContext(const CUContext&) = delete;
    CUContext& operator=(const CUContext&) = delete;

    CUContext(CUContext&& other);
    CUContext& operator=(CUContext&& other);

    operator CUcontext() const;

    void push() const;
    bool initialized() const;
  private:
    CUdevice device_;
    CUcontext context_;
    bool initialized_;
};

}
}
