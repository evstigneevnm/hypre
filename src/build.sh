#!/bin/bash
set -euo pipefail
set -x #echo on
MPI_ROOT_PATH=$1
CUDA_ROOT_PATH=$2
CUDA_ARCH_LIST=$3
HYPRE_ENABLE_GPU_AWARE_MPI=--enable-gpu-aware-mpi 
HYPRE_PRECISION_FLAG=""
HYPRE_DEBUG_FLAG=""
HYPRE_DEVICE_MODE=${7:-cuda}

if [[ "${STOKES_USE_CALLER_TOOLCHAIN:-0}" != "1" ]]
then
    # Conda exports compiler/linker variables and also provides MPI/Fortran
    # wrappers. Autoconf obeys both, so keep hypre on the requested MPI/CUDA
    # paths while preserving non-conda compiler shims from the caller's PATH.
    unset CC CXX CPP FC F77 F90 AR RANLIB LD
    unset CFLAGS CPPFLAGS CXXFLAGS FFLAGS FCFLAGS LDFLAGS LIBS
    unset build_alias host_alias target_alias
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
    CLEAN_PATH="$MPI_ROOT_PATH/bin:$CUDA_ROOT_PATH/bin"
    IFS=:
    for PATH_ENTRY in $PATH
    do
        case "$PATH_ENTRY" in
            *conda*|*anaconda*|"")
                ;;
            *)
                CLEAN_PATH="$CLEAN_PATH:$PATH_ENTRY"
                ;;
        esac
    done
    unset IFS
    export PATH="$CLEAN_PATH"
    export CC="$MPI_ROOT_PATH/bin/mpicc"
    export CXX="$MPI_ROOT_PATH/bin/mpic++"
    export AR="ar rc"
    export RANLIB=ranlib
fi

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
if [[ "${6:-}" == "no_cuda_aware" ]]
then
    echo "disabling --enable-gpu-aware-mpi"
    HYPRE_ENABLE_GPU_AWARE_MPI=""
    echo $HYPRE_ENABLE_GPU_AWARE_MPI
fi

HYPRE_DEVICE_CONFIG=()
if [[ "$HYPRE_DEVICE_MODE" == "cuda" ]]
then
    HYPRE_DEVICE_CONFIG=(--enable-cuda-streams "--with-cuda-home=$CUDA_ROOT_PATH" "--with-gpu-arch=$CUDA_ARCH_LIST" --enable-unified-memory)
elif [[ "$HYPRE_DEVICE_MODE" == "host" || "$HYPRE_DEVICE_MODE" == "cpu" ]]
then
    echo "Building CPU/OpenMP HYPRE without CUDA device backend"
    HYPRE_ENABLE_GPU_AWARE_MPI=""
else
    echo "Unknown HYPRE device mode: $HYPRE_DEVICE_MODE" >&2
    exit 2
fi

make clean || echo "Warning: make clean failed; continuing with a fresh configure"
./configure --with-MPI --enable-mixedint "--with-MPI-include=$MPI_ROOT_PATH/include" "--with-MPI-lib-dirs=$MPI_ROOT_PATH/lib" "${HYPRE_DEVICE_CONFIG[@]}" $HYPRE_PRECISION_FLAG $HYPRE_DEBUG_FLAG --with-openmp $HYPRE_ENABLE_GPU_AWARE_MPI 
make -j
if [[ "${STOKES_HYPRE_RUN_TEST:-1}" != "0" ]]
then
    cd test
    make ij -j
    ./ij
    cd ..
else
    echo "Skipping hypre runtime test because STOKES_HYPRE_RUN_TEST=0"
fi
