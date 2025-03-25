#!/bin/bash
set -x #echo on
MPI_ROOT_PATH=$1
CUDA_ROOT_PATH=$2
CUDA_ARCH_LIST=$3
echo "CUDA_ARCH_LIST=$CUDA_ARCH_LIST"
if [[ "$4" == "float" ]]
then
    echo "Using SINGLE PRECISION"
    HYPRE_PRECISION_FLAG=--enable-single
fi
if [[ "$5" == "debug" ]]
then
    echo "Using DEBUG VERSION"
    HYPRE_DEBUG_FLAG=--enable-debug
fi
make clean
./configure --with-MPI --with-openmp --enable-gpu-aware-mpi --with-MPI-include=$MPI_ROOT_PATH/include --with-MPI-lib-dirs=$MPI_ROOT_PATH/lib --enable-cuda-streams --with-cuda-home=$CUDA_ROOT_PATH --with-gpu-arch="$CUDA_ARCH_LIST" --enable-unified-memory $HYPRE_PRECISION_FLAG $HYPRE_DEBUG_FLAG
make -j
cd test
make ij -j
cd ..