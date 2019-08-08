#define CSC_MAT 0
#define CSR_MAT 1

SUNMatrix SUNSparseMatrix(sunindextype M, sunindextype N, sunindextype NNZ, int sparsetype);
SUNMatrix SUNSparseFromDenseMatrix(SUNMatrix A, realtype droptol, int sparsetype);
SUNMatrix SUNSparseFromBandMatrix(SUNMatrix A, realtype droptol, int sparsetype);
int SUNSparseMatrix_Realloc(SUNMatrix A);
int SUNSparseMatrix_Reallocate(SUNMatrix A, sunindextype NNZ);
void SUNSparseMatrix_Print(SUNMatrix A, FILE* outfile);

sunindextype SUNSparseMatrix_Rows(SUNMatrix A);
sunindextype SUNSparseMatrix_Columns(SUNMatrix A);
sunindextype SUNSparseMatrix_NNZ(SUNMatrix A);
sunindextype SUNSparseMatrix_NP(SUNMatrix A);
int SUNSparseMatrix_SparseType(SUNMatrix A);
realtype* SUNSparseMatrix_Data(SUNMatrix A);
sunindextype* SUNSparseMatrix_IndexValues(SUNMatrix A);
sunindextype* SUNSparseMatrix_IndexPointers(SUNMatrix A);

SUNMatrix_ID SUNMatGetID_Sparse(SUNMatrix A);
SUNMatrix SUNMatClone_Sparse(SUNMatrix A);
void SUNMatDestroy_Sparse(SUNMatrix A);
int SUNMatZero_Sparse(SUNMatrix A);
int SUNMatCopy_Sparse(SUNMatrix A, SUNMatrix B);
int SUNMatScaleAdd_Sparse(realtype c, SUNMatrix A, SUNMatrix B);
int SUNMatScaleAddI_Sparse(realtype c, SUNMatrix A);
int SUNMatMatvec_Sparse(SUNMatrix A, N_Vector x, N_Vector y);
int SUNMatSpace_Sparse(SUNMatrix A, long int *lenrw, long int *leniw);
