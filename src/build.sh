#!/bin/bash
set -x #echo on

PLATFORM="$1"  # Either HIP or CUDA
MPI_ROOT_PATH="$2"
eval "${PLATFORM}_ROOT_PATH=\$3"
eval "${PLATFORM}_ARCH_LIST=\$4"

if [[ "$5" == "float" ]]; then
    echo "Using SINGLE PRECISION"
    HYPRE_PRECISION_FLAG="--enable-single"
fi

if [[ "$6" == "debug" ]]; then
    echo "Using DEBUG VERSION"
    HYPRE_DEBUG_FLAG="--enable-debug"
fi

if [[ "$PLATFORM" == "CUDA" ]]; then
    echo "CUDA_ARCH_LIST=${CUDA_ARCH_LIST}"
    DEVICE_FLAGS="--with-cuda-home=${CUDA_ROOT_PATH} --with-gpu-arch="${CUDA_ARCH_LIST}" --enable-cuda-streams"
fi

if [[ "$PLATFORM" == "HIP" ]]; then
    echo "HIP_ARCH_LIST=${HIP_ARCH_LIST}"
    DEVICE_FLAGS="--with-hip --with-gpu-arch="${HIP_ARCH_LIST}""

    echo MPI:
    echo ${MPI_ROOT_PATH}
fi

make clean
./configure \
    --with-MPI \
    --with-openmp \
    --enable-gpu-aware-mpi \
    --enable-mixedint \
    --with-MPI-include="${MPI_ROOT_PATH}/include" \
    --with-MPI-lib-dirs="${MPI_ROOT_PATH}/lib" \
    --enable-unified-memory \
    ${DEVICE_FLAGS} \
    ${HYPRE_PRECISION_FLAG} \
    ${HYPRE_DEBUG_FLAG}

make -j
cd test || { echo "Failed to cd into test directory"; exit 1; }
make ij -j
./ij
cd ..
