/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * zzz_SMGInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGInitialize( MPI_Comm *comm )
{
   zzz_SMGData *smg_data;

   smg_data = zzz_CTAlloc(zzz_SMGData, 1);

   (smg_data -> comm)        = comm;
   (smg_data -> time_index)  = zzz_InitializeTiming("SMG");

   /* set defaults */
   (smg_data -> memory_use) = 0;
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> num_pre_relax)  = 1;
   (smg_data -> num_post_relax) = 1;
   (smg_data -> cdir) = 2;
   (smg_data -> ci) = 0;
   (smg_data -> fi) = 1;
   (smg_data -> cs) = 2;
   (smg_data -> fs) = 2;
   zzz_SetIndex((smg_data -> base_index), 0, 0, 0);
   zzz_SetIndex((smg_data -> base_stride), 1, 1, 1);

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGFinalize( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;

   int l;
   int ierr;

   if (smg_data)
   {
      if ((smg_data -> logging) > 0)
      {
         zzz_TFree(smg_data -> norms);
         zzz_TFree(smg_data -> rel_norms);
      }

      for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
      {
         zzz_SMGRelaxFinalize(smg_data -> relax_data_l[l]);
         zzz_SMGResidualFinalize(smg_data -> residual_data_l[l]);
         zzz_SMGRestrictFinalize(smg_data -> restrict_data_l[l]);
         zzz_SMGIntAddFinalize(smg_data -> intadd_data_l[l]);
      }
      zzz_SMGRelaxFinalize(smg_data -> relax_data_l[l]);
      zzz_TFree(smg_data -> relax_data_l);
      zzz_TFree(smg_data -> residual_data_l);
      zzz_TFree(smg_data -> restrict_data_l);
      zzz_TFree(smg_data -> intadd_data_l);
 
      zzz_FreeStructVector(smg_data -> tb_l[0]);
      zzz_FreeStructVector(smg_data -> tx_l[0]);
      for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
      {
         zzz_FreeStructGrid(smg_data -> grid_l[l+1]);
         zzz_FreeStructMatrix(smg_data -> A_l[l+1]);
         zzz_FreeStructMatrix(smg_data -> PT_l[l]);
         if (!zzz_StructMatrixSymmetric(smg_data -> A_l[0]))
            zzz_FreeStructMatrix(smg_data -> R_l[l]);
         zzz_FreeStructVector(smg_data -> b_l[l+1]);
         zzz_FreeStructVector(smg_data -> x_l[l+1]);
         zzz_FreeStructVectorShell(smg_data -> r_l[l+1]);
      }
      zzz_TFree(smg_data -> grid_l);
      zzz_TFree(smg_data -> A_l);
      zzz_TFree(smg_data -> PT_l);
      zzz_TFree(smg_data -> R_l);
      zzz_TFree(smg_data -> b_l);
      zzz_TFree(smg_data -> x_l);
      zzz_TFree(smg_data -> tb_l);
      zzz_TFree(smg_data -> tx_l);
 
      zzz_TFree(smg_data -> base_index_l);
      zzz_TFree(smg_data -> cindex_l);
      zzz_TFree(smg_data -> findex_l);
      zzz_TFree(smg_data -> base_stride_l);
      zzz_TFree(smg_data -> cstride_l);
      zzz_TFree(smg_data -> fstride_l);

      zzz_FinalizeTiming(smg_data -> time_index);
      zzz_TFree(smg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetMemoryUse
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetMemoryUse( void *smg_vdata,
                     int   memory_use )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> memory_use) = memory_use;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetTol
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetTol( void   *smg_vdata,
               double  tol       )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetMaxIter( void *smg_vdata,
                   int   max_iter  )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetZeroGuess( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> zero_guess) = 1;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetNumPreRelax
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetNumPreRelax( void *smg_vdata,
                    int   num_pre_relax )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> num_pre_relax) = max(num_pre_relax,1);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetNumPostRelax( void *smg_vdata,
                     int   num_post_relax )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetBase( void      *smg_vdata,
                zzz_Index  base_index,
                zzz_Index  base_stride )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          d;
   int          ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((smg_data -> base_index),  d) = zzz_IndexD(base_index,  d);
      zzz_IndexD((smg_data -> base_stride), d) = zzz_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetLogging
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetLogging( void *smg_vdata,
		   int   logging)
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
zzz_SMGGetNumIterations( void *smg_vdata,
                         int  *num_iterations )
{
   zzz_SMGData *smg_data = smg_vdata;
   int ierr;

   *num_iterations = (smg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGPrintLogging
 *--------------------------------------------------------------------------*/

int
zzz_SMGPrintLogging( void *smg_vdata,
                      int  myid)
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
   int          i;
   int          num_iterations  = (smg_data -> num_iterations);
   int          logging   = (smg_data -> logging);
   double      *norms     = (smg_data -> norms);
   double      *rel_norms = (smg_data -> rel_norms);

   
   if (myid == 0)
     {
       if (logging > 0)
	 {
	   for (i = 0; i < num_iterations; i++)
	     {
	       printf("Residual norm[%d] = %e   ",i,norms[i]);
	       printf("Relative residual norm[%d] = %e\n",i,rel_norms[i]);
	     }
	 }
     }
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
zzz_SMGGetFinalRelativeResidualNorm( void   *smg_vdata,
                                     double *relative_residual_norm )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = -1;
   int          num_iterations  = (smg_data -> num_iterations);
   int          logging   = (smg_data -> logging);
   double      *rel_norms = (smg_data -> rel_norms);

   
   if (logging > 0)
     {
       *relative_residual_norm = rel_norms[num_iterations-1];
       ierr = 0;
     }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetStructVectorConstantValues( zzz_StructVector *vector,
                                      double            values,
                                      zzz_SBoxArray    *sbox_array )
{
   int    ierr;

   zzz_Box          *v_data_box;

   int               vi;
   double           *vp;

   zzz_SBox         *sbox;
   zzz_Index         loop_size;
   zzz_IndexRef      start;
   zzz_IndexRef      stride;

   int               loopi, loopj, loopk;
   int               i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   zzz_ForSBoxI(i, sbox_array)
   {
      sbox  = zzz_SBoxArraySBox(sbox_array, i);
      start = zzz_SBoxIMin(sbox);
      stride = zzz_SBoxStride(sbox);

      v_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(vector), i);
      vp = zzz_StructVectorBoxData(vector, i);

      zzz_GetSBoxSize(sbox, loop_size);
      zzz_BoxLoop1(loopi, loopj, loopk, loop_size,
                   v_data_box, start, stride, vi,
                   {
                      vp[vi] = values;
                   });
   }

   return ierr;
}


