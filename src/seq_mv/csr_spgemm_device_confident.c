/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
           Perform SpMM with Row Nnz Upper Bound
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Numerical Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_numer( volatile HYPRE_Int
                                *HashKeys,      /* assumed to be initialized as all -1's */
                                volatile HYPRE_Complex *HashVals,      /* assumed to be initialized as all 0's */
                                HYPRE_Int      key,           /* assumed to be nonnegative */
                                HYPRE_Complex  val )
{
   HYPRE_Int j = 0;

#pragma unroll
   for (HYPRE_Int i = 0; i < SHMEM_HASH_SIZE; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (SHMEM_HASH_SIZE - 1);
      }
      else
      {
         j = HashFunc<SHMEM_HASH_SIZE, HASHTYPE>(key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int *)(HashKeys + j), -1, key);

      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd((HYPRE_Complex*)(HashVals + j), val);
         return j;
      }
   }

   return -1;
}

template <char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_numer(          HYPRE_Int      HashSize,
                                         volatile HYPRE_Int     *HashKeys,      /* assumed to be initialized as all -1's */
                                         volatile HYPRE_Complex *HashVals,      /* assumed to be initialized as all 0's */
                                         HYPRE_Int      key,           /* assumed to be nonnegative */
                                         HYPRE_Complex  val )
{
   HYPRE_Int j = 0;

#pragma unroll
   for (HYPRE_Int i = 0; i < HashSize; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (HashSize - 1);
      }
      else
      {
         j = HashFunc<HASHTYPE>(HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int *)(HashKeys + j), -1, key);

      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd((HYPRE_Complex*)(HashVals + j), val);
         return j;
      }
   }

   return -1;
}

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int GROUP_SIZE, bool HAS_GHASH>
static __device__ __forceinline__
void
hypre_spgemm_compute_row_numer(          HYPRE_Int      istart_a,
                                         HYPRE_Int      iend_a,
                                         const    HYPRE_Int     *ja,
                                         const    HYPRE_Complex *aa,
                                         const    HYPRE_Int     *ib,
                                         const    HYPRE_Int     *jb,
                                         const    HYPRE_Complex *ab,
                                         volatile HYPRE_Int     *s_HashKeys,
                                         volatile HYPRE_Complex *s_HashVals,
                                         HYPRE_Int      g_HashSize,
                                         HYPRE_Int     *g_HashKeys,
                                         HYPRE_Complex *g_HashVals)
{
   /* load column idx and values of row of A */
   for (HYPRE_Int i = istart_a + threadIdx.y; __any_sync(HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim.y)
   {
      HYPRE_Int     rowB = -1;
      HYPRE_Complex mult = 0.0;

      if (threadIdx.x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
         mult = read_only_load(aa + i);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, rowB, 0, blockDim.x);
      mult = __shfl_sync(HYPRE_WARP_FULL_MASK, mult, 0, blockDim.x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx.x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

      for (HYPRE_Int k = rowB_start + threadIdx.x; __any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim.x)
      {
         if (k < rowB_end)
         {
            const HYPRE_Int     k_idx = read_only_load(jb + k);
            const HYPRE_Complex k_val = read_only_load(ab + k) * mult;
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hypre_spgemm_hash_insert_numer<SHMEM_HASH_SIZE, HASHTYPE>
                            (s_HashKeys, s_HashVals, k_idx, k_val);

            if (HAS_GHASH && -1 == pos)
            {
               pos = hypre_spgemm_hash_insert_numer<HASHTYPE>
                     (g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val);
            }

            hypre_device_assert(pos != -1);
         }
      }
   }
}

template <HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool HAS_GHASH>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_copy_from_hash_into_C_row(          HYPRE_Int      lane_id,
                                                 volatile HYPRE_Int     *s_HashKeys,
                                                 volatile HYPRE_Complex *s_HashVals,
                                                 HYPRE_Int      ghash_size,
                                                 HYPRE_Int     *jg_start,
                                                 HYPRE_Complex *ag_start,
                                                 HYPRE_Int     *jc_start,
                                                 HYPRE_Complex *ac_start)
{
   HYPRE_Int j = 0;
   const HYPRE_Int STEP_SIZE = hypre_min(GROUP_SIZE, HYPRE_WARP_SIZE);

   /* copy shared memory hash table into C */
#pragma unroll
   for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += STEP_SIZE)
   {
      HYPRE_Int sum;
      HYPRE_Int key = s_HashKeys ? s_HashKeys[k] : -1;
      //const HYPRE_Int in = key != -1;
      HYPRE_Int pos = group_prefix_sum<HYPRE_Int, STEP_SIZE>(lane_id, (HYPRE_Int) (key != -1), sum);
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = s_HashVals[k];
      }
      j += sum;
   }

   if (HAS_GHASH)
   {
      /* copy global memory hash table into C */
#pragma unroll
      for (HYPRE_Int k = lane_id; __any_sync(HYPRE_WARP_FULL_MASK, k < ghash_size); k += STEP_SIZE)
      {
         HYPRE_Int sum;
         HYPRE_Int key = k < ghash_size ? jg_start[k] : -1;
         //const HYPRE_Int in = key != -1;
         HYPRE_Int pos = group_prefix_sum<HYPRE_Int, STEP_SIZE>(lane_id, (HYPRE_Int) (key != -1), sum);
         if (key != -1)
         {
            jc_start[j + pos] = key;
            ac_start[j + pos] = ag_start[k];
         }
         j += sum;
      }
   }

   return j;
}

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool FAILED_SYMBL, char HASHTYPE, bool HAS_GHASH, bool BINNED>
__global__ void
hypre_spgemm_numeric( const HYPRE_Int                   M, /* HYPRE_Int K, HYPRE_Int N, */
                      const HYPRE_Int*     __restrict__ rind,
                      const HYPRE_Int*     __restrict__ ia,
                      const HYPRE_Int*     __restrict__ ja,
                      const HYPRE_Complex* __restrict__ aa,
                      const HYPRE_Int*     __restrict__ ib,
                      const HYPRE_Int*     __restrict__ jb,
                      const HYPRE_Complex* __restrict__ ab,
                      const HYPRE_Int*     __restrict__ ic,
                      HYPRE_Int*     __restrict__ jc,
                      HYPRE_Complex* __restrict__ ac,
                      HYPRE_Int*     __restrict__ rc,
                      const HYPRE_Int*     __restrict__ ig,
                      HYPRE_Int*     __restrict__ jg,
                      HYPRE_Complex* __restrict__ ag)
{
   /* number of groups in the grid */
   volatile const HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   volatile const HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   volatile const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   volatile const HYPRE_Int lane_id = get_lane_id();
   /* lane id inside the warp */
   volatile const HYPRE_Int warp_lane_id = get_warp_lane_id();
   /* shared memory hash table */
   __shared__ volatile HYPRE_Int     s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   __shared__ volatile HYPRE_Complex s_HashVals[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this group */
   volatile HYPRE_Int     *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *group_s_HashVals = s_HashVals + group_id * SHMEM_HASH_SIZE;

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; __any_sync(HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      HYPRE_Int ii = -1;

      if (BINNED)
      {
         group_read<GROUP_SIZE>(rind + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                ii,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(ig + grid_group_id, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                istart_g, iend_g,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);

         /* size of global hash table allocated for this row
            (must be power of 2 and >= the actual size of the row of C) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
#pragma unroll
         for (HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
            ag[istart_g + k] = 0.0;
         }
      }

      /* initialize group's shared memory hash table */
      if (GROUP_SIZE >= HYPRE_WARP_SIZE || i < M)
      {
#pragma unroll
         for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
         {
            group_s_HashKeys[k] = -1;
            group_s_HashVals[k] = 0.0;
         }
      }

      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      HYPRE_Int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(ia + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                             istart_a, iend_a,
                             GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);

      /* work with two hash tables */
      hypre_spgemm_compute_row_numer<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH>(istart_a, iend_a,
                                                                                       ja, aa, ib, jb, ab,
                                                                                       group_s_HashKeys,
                                                                                       group_s_HashVals,
                                                                                       ghash_size,
                                                                                       jg + istart_g, ag + istart_g);

      if (GROUP_SIZE > HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      HYPRE_Int jsum = 0;

      /* copy results into the final C */
      if (get_warp_in_group_id<GROUP_SIZE>() == 0)
      {
         HYPRE_Int istart_c = 0;
#ifdef HYPRE_DEBUG
         HYPRE_Int iend_c = 0;
         group_read<GROUP_SIZE>(ic + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                istart_c, iend_c,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
#else
         group_read<GROUP_SIZE>(ic + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                istart_c,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
#endif

         jsum = hypre_spgemm_copy_from_hash_into_C_row<GROUP_SIZE, SHMEM_HASH_SIZE, HAS_GHASH>
                (lane_id,
                 GROUP_SIZE >= HYPRE_WARP_SIZE || i < M ? group_s_HashKeys : NULL,
                 group_s_HashVals,
                 ghash_size, jg + istart_g, ag + istart_g,
                 jc + istart_c, ac + istart_c);

#if defined(HYPRE_DEBUG)
         if (FAILED_SYMBL)
         {
            hypre_device_assert(istart_c + jsum <= iend_c);
         }
         else
         {
            hypre_device_assert(istart_c + jsum == iend_c);
         }
#endif
      }

      if (GROUP_SIZE > HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      if (FAILED_SYMBL)
      {
         /* when symb mult was failed, save (exact) row nnz */
         if ((GROUP_SIZE >= HYPRE_WARP_SIZE || i < M) && lane_id == 0)
         {
            rc[ii] = jsum;
         }
      }
   }
}

template <HYPRE_Int GROUP_SIZE>
__global__ void
hypre_spgemm_copy_from_Cext_into_C( HYPRE_Int      M,
                                    HYPRE_Int     *ix,
                                    HYPRE_Int     *jx,
                                    HYPRE_Complex *ax,
                                    HYPRE_Int     *ic,
                                    HYPRE_Int     *jc,
                                    HYPRE_Complex *ac )
{
   /* number of groups in the grid */
   const HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   const HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   const HYPRE_Int lane_id = get_lane_id();
   /* lane id inside the warp */
   const HYPRE_Int warp_lane_id = get_warp_lane_id();

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; __any_sync(HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      HYPRE_Int istart_c = 0, iend_c = 0, istart_x = 0;

      /* start/end position in C and X */
      group_read<GROUP_SIZE>(ic + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                             istart_c, iend_c,
                             GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
#if defined(HYPRE_DEBUG)
      HYPRE_Int iend_x = 0;
      group_read<GROUP_SIZE>(ix + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                             istart_x, iend_x,
                             GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
      hypre_device_assert(iend_c - istart_c <= iend_x - istart_x);
#else
      group_read<GROUP_SIZE>(ix + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                             istart_x,
                             GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
#endif
      const HYPRE_Int p = istart_x - istart_c;
      for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += GROUP_SIZE)
      {
         jc[k] = jx[k + p];
         ac[k] = ax[k + p];
      }
   }
}

template <HYPRE_Int GROUP_SIZE>
HYPRE_Int
hypreDevice_CSRSpGemmNumerPostCopy( HYPRE_Int       m,
                                    HYPRE_Int      *d_rc,
                                    HYPRE_Int      *nnzC,
                                    HYPRE_Int     **d_ic,
                                    HYPRE_Int     **d_jc,
                                    HYPRE_Complex **d_c)
{
   HYPRE_Int nnzC_new = hypreDevice_IntegerReduceSum(m, d_rc);

   hypre_assert(nnzC_new <= *nnzC && nnzC_new >= 0);

   if (nnzC_new < *nnzC)
   {
      HYPRE_Int     *d_ic_new = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *d_jc_new;
      HYPRE_Complex *d_c_new;

      /* alloc final C */
      hypre_create_ija(m, NULL, d_rc, d_ic_new, &d_jc_new, &d_c_new, &nnzC_new);

#ifdef HYPRE_SPGEMM_PRINTF
      printf("%s[%d]: Post Copy: new nnzC %d\n", __func__, __LINE__, nnzC_new);
#endif

      /* copy to the final C */
      const HYPRE_Int num_groups_per_block  = hypre_min(16 * HYPRE_WARP_SIZE / GROUP_SIZE, 64);
      dim3 bDim(GROUP_SIZE, 1, num_groups_per_block);
      dim3 gDim( (m + bDim.z - 1) / bDim.z );

      HYPRE_CUDA_LAUNCH( (hypre_spgemm_copy_from_Cext_into_C<GROUP_SIZE>), gDim, bDim,
                         m, *d_ic, *d_jc, *d_c, d_ic_new, d_jc_new, d_c_new );

      hypre_TFree(*d_ic, HYPRE_MEMORY_DEVICE);
      hypre_TFree(*d_jc, HYPRE_MEMORY_DEVICE);
      hypre_TFree(*d_c,  HYPRE_MEMORY_DEVICE);

      *d_ic = d_ic_new;
      *d_jc = d_jc_new;
      *d_c  = d_c_new;
      *nnzC = nnzC_new;
   }

   return hypre_error_flag;
}


/* SpGeMM with Rownnz/Upper bound */
template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool NEED_GHASH, bool BINNED>
HYPRE_Int
hypre_spgemm_numerical_with_rownnz( HYPRE_Int      m,
                                    HYPRE_Int     *row_ind,
                                    HYPRE_Int      k,
                                    HYPRE_Int      n,
                                    HYPRE_Int      EXACT_ROWNNZ,
                                    char           hash_type,
                                    HYPRE_Int     *d_ia,
                                    HYPRE_Int     *d_ja,
                                    HYPRE_Complex *d_a,
                                    HYPRE_Int     *d_ib,
                                    HYPRE_Int     *d_jb,
                                    HYPRE_Complex *d_b,
                                    HYPRE_Int     *d_rc,
                                    HYPRE_Int     *d_ic,
                                    HYPRE_Int     *d_jc,
                                    HYPRE_Complex *d_c )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_CUDA)
   const HYPRE_Int num_groups_per_block  = hypre_min(16 * HYPRE_WARP_SIZE / GROUP_SIZE, 64);
   const HYPRE_Int BDIMX                 = 2;
#elif defined(HYPRE_USING_HIP)
   const HYPRE_Int num_groups_per_block  = hypre_min(16 * HYPRE_WARP_SIZE / GROUP_SIZE, 64);
   const HYPRE_Int BDIMX                 = 4;
#endif
   const HYPRE_Int num_threads_per_block = num_groups_per_block * GROUP_SIZE;
   const HYPRE_Int BDIMY                 = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
   hypre_assert(bDim.x * bDim.y == GROUP_SIZE);
   // total number of threads used: a group works on a row
   HYPRE_Int num_threads = hypre_min((size_t) m * GROUP_SIZE, HYPRE_MAX_NUM_WARPS * HYPRE_WARP_SIZE);
   // grid dimension (number of blocks)
   dim3 gDim( (num_threads + num_threads_per_block - 1) / num_threads_per_block );
   // number of active groups
   HYPRE_Int num_act_groups = hypre_min(bDim.z * gDim.x, m);

   /* ---------------------------------------------------------------------------
    * build hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int     *d_ghash_i = NULL;
   HYPRE_Int     *d_ghash_j = NULL;
   HYPRE_Complex *d_ghash_a = NULL;
   HYPRE_Int      ghash_size = 0;

   if (NEED_GHASH)
   {
      hypre_SpGemmCreateGlobalHashTable(m, row_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                        &d_ghash_i, &d_ghash_j, &d_ghash_a, &ghash_size);
   }

#ifdef HYPRE_SPGEMM_PRINTF
   printf("%s[%d]: HASH %c, SHMEM_HASH_SIZE %d, GROUP_SIZE %d, EXACT_ROWNNZ %d, ghash %p size %d\n",
          __func__, __LINE__,
          hash_type, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, d_ghash_i, ghash_size);
   printf("%s[%d]: kernel spec [%d %d %d] x [%d %d %d]\n", __func__, __LINE__, gDim.x, gDim.y, gDim.z,
          bDim.x, bDim.y, bDim.z);
#endif

   /* ---------------------------------------------------------------------------
    * numerical multiplication:
    * ---------------------------------------------------------------------------*/
   if (EXACT_ROWNNZ)
   {
      if (NEED_GHASH && ghash_size)
      {
         HYPRE_CUDA_LAUNCH ( (
                                hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, false, 'D', true, BINNED>),
                             gDim, bDim, /* shmem_size, */
                             m, row_ind, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, NULL,
                             d_ghash_i, d_ghash_j, d_ghash_a );
      }
      else
      {
         HYPRE_CUDA_LAUNCH ( (
                                hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, false, 'D', false, BINNED>),
                             gDim, bDim, /* shmem_size, */
                             m, row_ind, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, NULL,
                             d_ghash_i, d_ghash_j, d_ghash_a );
      }
   }
   else
   {
      if (NEED_GHASH && ghash_size)
      {
         HYPRE_CUDA_LAUNCH ( (
                                hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, 'D', true, BINNED>),
                             gDim, bDim, /* shmem_size, */
                             m, row_ind, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rc,
                             d_ghash_i, d_ghash_j, d_ghash_a );
      }
      else
      {
         HYPRE_CUDA_LAUNCH ( (
                                hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, 'D', false, BINNED>),
                             gDim, bDim, /* shmem_size, */
                             m, row_ind, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rc,
                             d_ghash_i, d_ghash_j, d_ghash_a );
      }
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_a, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperbound( HYPRE_Int       m,
                                                HYPRE_Int       k,
                                                HYPRE_Int       n,
                                                HYPRE_Int      *d_ia,
                                                HYPRE_Int      *d_ja,
                                                HYPRE_Complex  *d_a,
                                                HYPRE_Int      *d_ib,
                                                HYPRE_Int      *d_jb,
                                                HYPRE_Complex  *d_b,
                                                HYPRE_Int      *d_rc,         /* input: nnz (upper bound) of each row */
                                                HYPRE_Int       exact_rownnz, /* if d_rc is exact */
                                                HYPRE_Int     **d_ic_out,
                                                HYPRE_Int     **d_jc_out,
                                                HYPRE_Complex **d_c_out,
                                                HYPRE_Int      *nnzC_out )
{
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmNumerBound");
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());
   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_NUMER_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_WARP_SIZE;

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

#ifdef HYPRE_SPGEMM_PRINTF
   HYPRE_Int max_rc = HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, 0,      thrust::maximum<HYPRE_Int>());
   HYPRE_Int min_rc = HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, max_rc, thrust::minimum<HYPRE_Int>());
   printf("%s[%d]: max RC %d, min RC %d\n", __func__, __LINE__, max_rc, min_rc);
#endif

   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC = -1;

   hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

#ifdef HYPRE_SPGEMM_PRINTF
   printf("%s[%d]: nnzC %d\n", __func__, __LINE__, nnzC);
#endif

   /* RL Note: even with exact rownnz, still may need global hash, since shared hash has different size from symbol. */
   hypre_spgemm_numerical_with_rownnz<SHMEM_HASH_SIZE, GROUP_SIZE, true, false>
   (m, NULL, k, n, exact_rownnz, hash_type, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<GROUP_SIZE>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#define HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(b, SHMEM_HASH_SIZE, GROUP_SIZE, GHASH)                                  \
{                                                                                                                         \
   const HYPRE_Int p = h_bin_ptr[b - 1];                                                                                  \
   const HYPRE_Int q = h_bin_ptr[b];                                                                                      \
   const HYPRE_Int bs = q - p;                                                                                            \
   printf("!! %d rows\n", bs);                                                                                            \
   if (bs)                                                                                                                \
   {                                                                                                                      \
      hypre_spgemm_numerical_with_rownnz<SHMEM_HASH_SIZE, GROUP_SIZE, GHASH, true>                                        \
      (bs, d_rc_indice + p, k, n, exact_rownnz, hash_type, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);      \
   }                                                                                                                      \
}

HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned( HYPRE_Int       m,
                                                      HYPRE_Int       k,
                                                      HYPRE_Int       n,
                                                      HYPRE_Int      *d_ia,
                                                      HYPRE_Int      *d_ja,
                                                      HYPRE_Complex  *d_a,
                                                      HYPRE_Int      *d_ib,
                                                      HYPRE_Int      *d_jb,
                                                      HYPRE_Complex  *d_b,
                                                      HYPRE_Int      *d_rc,         /* input: nnz (upper bound) of each row */
                                                      HYPRE_Int       exact_rownnz, /* if d_rc is exact */
                                                      HYPRE_Int     **d_ic_out,
                                                      HYPRE_Int     **d_jc_out,
                                                      HYPRE_Complex **d_c_out,
                                                      HYPRE_Int      *nnzC_out )
{
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmNumerBinned");
#endif

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1, t2;
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC = -1;

   hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

   HYPRE_Int *d_rc_indice = d_rc + m;
   HYPRE_Int *d_bin_ptr = hypre_TAlloc(HYPRE_Int, HYPRE_SPGEMM_NBIN + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int  h_bin_ptr[HYPRE_SPGEMM_NBIN + 1];
   const HYPRE_Int s = 12;

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   /* assume there are no more than 127 = 2^7-1 bins, which should be enough */
   char *d_bin_key = hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      d_rc,
                      d_rc + m,
                      d_bin_key,
                      spgemm_bin_op<HYPRE_Int, s>() );

   HYPRE_THRUST_CALL( sequence, d_rc_indice, d_rc_indice + m);
   HYPRE_THRUST_CALL( stable_sort_by_key, d_bin_key, d_bin_key + m, d_rc_indice );

   HYPRE_THRUST_CALL( lower_bound,
                      d_bin_key,
                      d_bin_key + m,
                      thrust::make_counting_iterator(1),
                      thrust::make_counting_iterator(HYPRE_SPGEMM_NBIN + 2),
                      d_bin_ptr );

   hypre_TFree(d_bin_key, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(h_bin_ptr, d_bin_ptr, HYPRE_Int, HYPRE_SPGEMM_NBIN + 1, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);
   hypre_assert(h_bin_ptr[HYPRE_SPGEMM_NBIN] == m);

   hypre_TFree(d_bin_ptr, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_SyncCudaComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   printf("%s[%d]: binning time %f\n", __func__, __LINE__, t2);
#endif

   /*
   HYPRE_Int *d_tmp = hypre_TAlloc(HYPRE_Int, HYPRE_SPGEMM_NBIN, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *h_tmp = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_HOST);
   HYPRE_THRUST_CALL( transform,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(HYPRE_SPGEMM_NBIN),
                      d_tmp,
                      ((1 << _1) >> 1) * s );

   hypre_TMemcpy(h_tmp, d_tmp, HYPRE_Int, HYPRE_SPGEMM_NBIN, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   for (int i = 0; i < HYPRE_SPGEMM_NBIN; i++) printf("%d ", h_tmp[i]); printf("\n");

   HYPRE_Int *h_rc = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(h_rc, d_rc_copy,       HYPRE_Int, m, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   for (int i = 0; i < m; i++) printf("%d(%d) ", h_rc[i], i); printf("\n");
   for (int i = 0; i < HYPRE_SPGEMM_NBIN; i++) printf("%d ", h_bin_ptr[i]); printf("\n");
   */

   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 1,  HYPRE_SPGEMM_NUMER_HASH_SIZE / 16,
                                              HYPRE_WARP_SIZE / 16, false);  /* 16,      2 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 2,  HYPRE_SPGEMM_NUMER_HASH_SIZE /  8,
                                              HYPRE_WARP_SIZE /  8, false);  /* 32,      4 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 3,  HYPRE_SPGEMM_NUMER_HASH_SIZE /  4,
                                              HYPRE_WARP_SIZE /  4, false);  /* 64,      8 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 4,  HYPRE_SPGEMM_NUMER_HASH_SIZE /  2,
                                              HYPRE_WARP_SIZE /  2, false);  /* 128,    16 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 5,  HYPRE_SPGEMM_NUMER_HASH_SIZE, HYPRE_WARP_SIZE,
                                              false);  /* 256,    32 */
#if 0
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 6,  HYPRE_SPGEMM_NUMER_HASH_SIZE *  2,
                                              HYPRE_WARP_SIZE *  2, false);  /* 512,    64 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 7,  HYPRE_SPGEMM_NUMER_HASH_SIZE *  4,
                                              HYPRE_WARP_SIZE *  4, false);  /* 1024,  128 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 8,  HYPRE_SPGEMM_NUMER_HASH_SIZE *  8,
                                              HYPRE_WARP_SIZE *  8, false);  /* 2048,  256 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 9,  HYPRE_SPGEMM_NUMER_HASH_SIZE * 16,
                                              HYPRE_WARP_SIZE * 16, false);  /* 4096,  512 */
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(10,  HYPRE_SPGEMM_NUMER_HASH_SIZE * 32,
                                             HYPRE_WARP_SIZE * 32, true);   /* 8192, 1024 */
#endif

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<HYPRE_WARP_SIZE>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

