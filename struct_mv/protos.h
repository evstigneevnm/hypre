#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* ZZZ_struct_grid.c */
ZZZ_StructGrid ZZZ_NewStructGrid P((MPI_Comm *comm , int dim ));
void ZZZ_FreeStructGrid P((ZZZ_StructGrid grid ));
void ZZZ_SetStructGridExtents P((ZZZ_StructGrid grid , int *ilower , int *iupper ));
void ZZZ_AssembleStructGrid P((ZZZ_StructGrid grid ));

/* ZZZ_struct_matrix.c */
ZZZ_StructMatrix ZZZ_NewStructMatrix P((MPI_Comm *comm , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_InitializeStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_SetStructMatrixValues P((ZZZ_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_SetStructMatrixCoeffs P((ZZZ_StructMatrix matrix , int *grid_index , double *values ));
int ZZZ_SetStructMatrixBoxValues P((ZZZ_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_AssembleStructMatrix P((ZZZ_StructMatrix matrix ));
void ZZZ_SetStructMatrixNumGhost P((ZZZ_StructMatrix matrix , int *num_ghost ));
ZZZ_StructGrid ZZZ_StructMatrixGrid P((ZZZ_StructMatrix matrix ));
void ZZZ_SetStructMatrixSymmetric P((ZZZ_StructMatrix matrix , int symmetric ));
void ZZZ_PrintStructMatrix P((char *filename , ZZZ_StructMatrix matrix , int all ));

/* ZZZ_struct_stencil.c */
ZZZ_StructStencil ZZZ_NewStructStencil P((int dim , int size ));
void ZZZ_SetStructStencilElement P((ZZZ_StructStencil stencil , int element_index , int *offset ));
void ZZZ_FreeStructStencil P((ZZZ_StructStencil stencil ));

/* ZZZ_struct_vector.c */
ZZZ_StructVector ZZZ_NewStructVector P((MPI_Comm *comm , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructVector P((ZZZ_StructVector struct_vector ));
int ZZZ_InitializeStructVector P((ZZZ_StructVector vector ));
int ZZZ_SetStructVectorValues P((ZZZ_StructVector vector , int *grid_index , double values ));
int ZZZ_GetStructVectorValues P((ZZZ_StructVector vector , int *grid_index , double *values ));
int ZZZ_SetStructVectorBoxValues P((ZZZ_StructVector vector , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_AssembleStructVector P((ZZZ_StructVector vector ));
void ZZZ_PrintStructVector P((char *filename , ZZZ_StructVector vector , int all ));

/* box.c */
int zzz_SetIndex P((zzz_Index index , int ix , int iy , int iz ));
int zzz_CopyIndex P((zzz_Index index1 , zzz_Index index2 ));
zzz_Box *zzz_NewBox P((zzz_Index imin , zzz_Index imax ));
zzz_BoxArray *zzz_NewBoxArray P((void ));
zzz_BoxArrayArray *zzz_NewBoxArrayArray P((int size ));
void zzz_FreeBox P((zzz_Box *box ));
void zzz_FreeBoxArrayShell P((zzz_BoxArray *box_array ));
void zzz_FreeBoxArray P((zzz_BoxArray *box_array ));
void zzz_FreeBoxArrayArrayShell P((zzz_BoxArrayArray *box_array_array ));
void zzz_FreeBoxArrayArray P((zzz_BoxArrayArray *box_array_array ));
zzz_Box *zzz_DuplicateBox P((zzz_Box *box ));
zzz_BoxArray *zzz_DuplicateBoxArray P((zzz_BoxArray *box_array ));
zzz_BoxArrayArray *zzz_DuplicateBoxArrayArray P((zzz_BoxArrayArray *box_array_array ));
void zzz_AppendBox P((zzz_Box *box , zzz_BoxArray *box_array ));
void zzz_DeleteBox P((zzz_BoxArray *box_array , int index ));
void zzz_AppendBoxArray P((zzz_BoxArray *box_array_0 , zzz_BoxArray *box_array_1 ));
void zzz_AppendBoxArrayArray P((zzz_BoxArrayArray *box_array_array_0 , zzz_BoxArrayArray *box_array_array_1 ));
int zzz_GetBoxSize P((zzz_Box *box , zzz_Index size ));
void zzz_CopyBoxArrayData P((zzz_BoxArray *box_array_in , zzz_BoxArray *data_space_in , int num_values_in , double *data_in , zzz_BoxArray *box_array_out , zzz_BoxArray *data_space_out , int num_values_out , double *data_out ));

/* box_algebra.c */
zzz_Box *zzz_IntersectBoxes P((zzz_Box *box1 , zzz_Box *box2 ));
zzz_BoxArray *zzz_IntersectBoxArrays P((zzz_BoxArray *box_array1 , zzz_BoxArray *box_array2 ));
zzz_BoxArray *zzz_SubtractBoxes P((zzz_Box *box1 , zzz_Box *box2 ));
zzz_BoxArray *zzz_UnionBoxArray P((zzz_BoxArray *boxes ));

/* communication.c */
void zzz_GetCommInfo P((zzz_BoxArrayArray **send_boxes_ptr , zzz_BoxArrayArray **recv_boxes_ptr , int ***send_box_ranks_ptr , int ***recv_box_ranks_ptr , zzz_StructGrid *grid , zzz_StructStencil *stencil ));
void zzz_GetSBoxType P((zzz_SBox *comm_sbox , zzz_Box *data_box , int num_values , MPI_Datatype *comm_sbox_type ));
zzz_CommPkg *zzz_NewCommPkg P((zzz_SBoxArrayArray *send_sboxes , zzz_SBoxArrayArray *recv_sboxes , int **send_box_ranks , int **recv_box_ranks , zzz_StructGrid *grid , zzz_BoxArray *data_space , int num_values ));
void zzz_FreeCommPkg P((zzz_CommPkg *comm_pkg ));
zzz_CommHandle *zzz_NewCommHandle P((int num_requests , MPI_Request *requests ));
void zzz_FreeCommHandle P((zzz_CommHandle *comm_handle ));
zzz_CommHandle *zzz_InitializeCommunication P((zzz_CommPkg *comm_pkg , double *data ));
void zzz_FinalizeCommunication P((zzz_CommHandle *comm_handle ));

/* computation.c */
void zzz_GetComputeInfo P((zzz_BoxArrayArray **send_boxes_ptr , zzz_BoxArrayArray **recv_boxes_ptr , int ***send_box_ranks_ptr , int ***recv_box_ranks_ptr , zzz_BoxArrayArray **indt_boxes_ptr , zzz_BoxArrayArray **dept_boxes_ptr , zzz_StructGrid *grid , zzz_StructStencil *stencil ));
zzz_ComputePkg *zzz_NewComputePkg P((zzz_SBoxArrayArray *send_sboxes , zzz_SBoxArrayArray *recv_sboxes , int **send_box_ranks , int **recv_box_ranks , zzz_SBoxArrayArray *indt_sboxes , zzz_SBoxArrayArray *dept_sboxes , zzz_StructGrid *grid , zzz_BoxArray *data_space , int num_values ));
void zzz_FreeComputePkg P((zzz_ComputePkg *compute_pkg ));
zzz_CommHandle *zzz_InitializeIndtComputations P((zzz_ComputePkg *compute_pkg , double *data ));
void zzz_FinalizeIndtComputations P((zzz_CommHandle *comm_handle ));

/* create_2d_laplacian.c */
int main P((int argc , char *argv []));

/* create_3d_laplacian.c */
int main P((int argc , char *argv []));

/* driver.c */
int main P((int argc , char *argv []));

/* driver_internal.c */
int main P((int argc , char *argv []));

/* grow.c */
zzz_BoxArray *zzz_GrowBoxByStencil P((zzz_Box *box , zzz_StructStencil *stencil , int transpose ));
zzz_BoxArrayArray *zzz_GrowBoxArrayByStencil P((zzz_BoxArray *box_array , zzz_StructStencil *stencil , int transpose ));

/* matrix_interface_struct.c */

/* neighbors.c */
void zzz_FindBoxNeighbors P((zzz_BoxArray *boxes , zzz_BoxArray *all_boxes , zzz_StructStencil *stencil , int transpose , zzz_BoxArray **neighbors_ptr , int **neighbor_ranks_ptr ));
void zzz_FindBoxApproxNeighbors P((zzz_BoxArray *boxes , zzz_BoxArray *all_boxes , zzz_StructStencil *stencil , int transpose , zzz_BoxArray **neighbors_ptr , int **neighbor_ranks_ptr ));

/* one_to_many.c */
int main P((int argc , char *argv []));

/* one_to_many_vector.c */
int main P((int argc , char *argv []));

/* project.c */
zzz_SBox *zzz_ProjectBox P((zzz_Box *box , zzz_Index index , zzz_Index stride ));
zzz_SBoxArray *zzz_ProjectBoxArray P((zzz_BoxArray *box_array , zzz_Index index , zzz_Index stride ));
zzz_SBoxArrayArray *zzz_ProjectBoxArrayArray P((zzz_BoxArrayArray *box_array_array , zzz_Index index , zzz_Index stride ));
zzz_SBoxArrayArray *zzz_ProjectRBPoint P((zzz_BoxArrayArray *box_array_array , zzz_Index rb [4 ]));

/* sbox.c */
zzz_SBox *zzz_NewSBox P((zzz_Box *box , zzz_Index stride ));
zzz_SBoxArray *zzz_NewSBoxArray P((void ));
zzz_SBoxArrayArray *zzz_NewSBoxArrayArray P((int size ));
void zzz_FreeSBox P((zzz_SBox *sbox ));
void zzz_FreeSBoxArrayShell P((zzz_SBoxArray *sbox_array ));
void zzz_FreeSBoxArray P((zzz_SBoxArray *sbox_array ));
void zzz_FreeSBoxArrayArrayShell P((zzz_SBoxArrayArray *sbox_array_array ));
void zzz_FreeSBoxArrayArray P((zzz_SBoxArrayArray *sbox_array_array ));
zzz_SBox *zzz_DuplicateSBox P((zzz_SBox *sbox ));
zzz_SBoxArray *zzz_DuplicateSBoxArray P((zzz_SBoxArray *sbox_array ));
zzz_SBoxArrayArray *zzz_DuplicateSBoxArrayArray P((zzz_SBoxArrayArray *sbox_array_array ));
zzz_SBox *zzz_ConvertToSBox P((zzz_Box *box ));
zzz_SBoxArray *zzz_ConvertToSBoxArray P((zzz_BoxArray *box_array ));
zzz_SBoxArrayArray *zzz_ConvertToSBoxArrayArray P((zzz_BoxArrayArray *box_array_array ));
void zzz_AppendSBox P((zzz_SBox *sbox , zzz_SBoxArray *sbox_array ));
void zzz_DeleteSBox P((zzz_SBoxArray *sbox_array , int index ));
void zzz_AppendSBoxArray P((zzz_SBoxArray *sbox_array_0 , zzz_SBoxArray *sbox_array_1 ));
void zzz_AppendSBoxArrayArray P((zzz_SBoxArrayArray *sbox_array_array_0 , zzz_SBoxArrayArray *sbox_array_array_1 ));
int zzz_GetSBoxSize P((zzz_SBox *sbox , zzz_Index size ));

/* struct_axpy.c */
int zzz_StructAxpy P((double alpha , zzz_StructVector *x , zzz_StructVector *y ));

/* struct_copy.c */
int zzz_StructCopy P((zzz_StructVector *x , zzz_StructVector *y ));

/* struct_grid.c */
zzz_StructGrid *zzz_NewStructGrid P((MPI_Comm *comm , int dim ));
zzz_StructGrid *zzz_NewAssembledStructGrid P((MPI_Comm *comm , int dim , zzz_BoxArray *all_boxes , int *processes ));
void zzz_FreeStructGrid P((zzz_StructGrid *grid ));
void zzz_SetStructGridExtents P((zzz_StructGrid *grid , zzz_Index ilower , zzz_Index iupper ));
void zzz_AssembleStructGrid P((zzz_StructGrid *grid ));
void zzz_PrintStructGrid P((FILE *file , zzz_StructGrid *grid ));
zzz_StructGrid *zzz_ReadStructGrid P((MPI_Comm *comm , FILE *file ));

/* struct_innerprod.c */
double zzz_StructInnerProd P((zzz_StructVector *x , zzz_StructVector *y ));

/* struct_io.c */
void zzz_PrintBoxArrayData P((FILE *file , zzz_BoxArray *box_array , zzz_BoxArray *data_space , int num_values , double *data ));
void zzz_ReadBoxArrayData P((FILE *file , zzz_BoxArray *box_array , zzz_BoxArray *data_space , int num_values , double *data ));

/* struct_matrix.c */
double *zzz_StructMatrixExtractPointerByIndex P((zzz_StructMatrix *matrix , int b , zzz_Index index ));
zzz_StructMatrix *zzz_NewStructMatrix P((MPI_Comm *comm , zzz_StructGrid *grid , zzz_StructStencil *user_stencil ));
int zzz_FreeStructMatrix P((zzz_StructMatrix *matrix ));
int zzz_InitializeStructMatrixShell P((zzz_StructMatrix *matrix ));
void zzz_InitializeStructMatrixData P((zzz_StructMatrix *matrix , double *data ));
int zzz_InitializeStructMatrix P((zzz_StructMatrix *matrix ));
int zzz_SetStructMatrixValues P((zzz_StructMatrix *matrix , zzz_Index grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int zzz_SetStructMatrixBoxValues P((zzz_StructMatrix *matrix , zzz_Box *value_box , int num_stencil_indices , int *stencil_indices , double *values ));
int zzz_AssembleStructMatrix P((zzz_StructMatrix *matrix ));
void zzz_SetStructMatrixNumGhost P((zzz_StructMatrix *matrix , int *num_ghost ));
void zzz_PrintStructMatrix P((char *filename , zzz_StructMatrix *matrix , int all ));
zzz_StructMatrix *zzz_ReadStructMatrix P((MPI_Comm *comm , char *filename , int *num_ghost ));

/* struct_matrix_mask.c */
zzz_StructMatrix *zzz_NewStructMatrixMask P((zzz_StructMatrix *matrix , int num_stencil_indices , int *stencil_indices ));
int zzz_FreeStructMatrixMask P((zzz_StructMatrix *mask ));

/* struct_matvec.c */
void *zzz_StructMatvecInitialize P((void ));
int zzz_StructMatvecSetup P((void *matvec_vdata , zzz_StructMatrix *A , zzz_StructVector *x ));
int zzz_StructMatvecCompute P((void *matvec_vdata , double alpha , double beta , zzz_StructVector *y ));
int zzz_StructMatvecFinalize P((void *matvec_vdata ));
int zzz_StructMatvec P((double alpha , zzz_StructMatrix *A , zzz_StructVector *x , double beta , zzz_StructVector *y ));

/* struct_scale.c */
int zzz_StructScale P((double alpha , zzz_StructVector *y ));

/* struct_stencil.c */
zzz_StructStencil *zzz_NewStructStencil P((int dim , int size , zzz_Index *shape ));
void zzz_FreeStructStencil P((zzz_StructStencil *stencil ));
int zzz_StructStencilElementRank P((zzz_StructStencil *stencil , zzz_Index stencil_element ));
int zzz_SymmetrizeStructStencil P((zzz_StructStencil *stencil , zzz_StructStencil **symm_stencil_ptr , int **symm_elements_ptr ));

/* struct_vector.c */
zzz_StructVector *zzz_NewStructVector P((MPI_Comm *comm , zzz_StructGrid *grid ));
int zzz_FreeStructVectorShell P((zzz_StructVector *vector ));
int zzz_FreeStructVector P((zzz_StructVector *vector ));
int zzz_InitializeStructVectorShell P((zzz_StructVector *vector ));
void zzz_InitializeStructVectorData P((zzz_StructVector *vector , double *data ));
int zzz_InitializeStructVector P((zzz_StructVector *vector ));
int zzz_SetStructVectorValues P((zzz_StructVector *vector , zzz_Index grid_index , double values ));
int zzz_GetStructVectorValues P((zzz_StructVector *vector , zzz_Index grid_index , double *values ));
int zzz_SetStructVectorBoxValues P((zzz_StructVector *vector , zzz_Box *value_box , double *values ));
int zzz_SetStructVectorConstantValues P((zzz_StructVector *vector , double values ));
int zzz_ClearStructVectorGhostValues P((zzz_StructVector *vector ));
int zzz_ClearStructVectorAllValues P((zzz_StructVector *vector ));
int zzz_AssembleStructVector P((zzz_StructVector *vector ));
void zzz_SetStructVectorNumGhost P((zzz_StructVector *vector , int *num_ghost ));
void zzz_PrintStructVector P((char *filename , zzz_StructVector *vector , int all ));
zzz_StructVector *zzz_ReadStructVector P((MPI_Comm *comm , char *filename , int *num_ghost ));

#undef P
