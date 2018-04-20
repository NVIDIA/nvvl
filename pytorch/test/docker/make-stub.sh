#!/bin/bash

# First find the versioned library
PATHS=$(sed 's/:/ /g' <<< $LD_LIBRARY_PATH)
PATHS+=$(ldconfig -v 2>/dev/null | grep -v ^$'\t' | sed 's/:/ /')
for p in $PATHS; do
    LIBPATH=${p}/libnvcuvid.so.1
    if [[ -f $LIBPATH ]]; then
        break
    fi
done

nm -D --defined-only $LIBPATH |
    awk '{print $3}' |
    grep -vE '^__' |
    sed 's/.*/CUresult &() { return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND; }/' |
    cat <(echo '#include <cuda.h>') - |
    gcc -shared -I/usr/local/cuda/include -Wl,--soname=libnvcuvid.so.1 -nostdlib -o libnvcuvid.so -x c -

strip libnvcuvid.so
