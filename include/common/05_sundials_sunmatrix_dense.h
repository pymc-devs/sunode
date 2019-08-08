SUNMatrix SUNDenseMatrix(sunindextype M, sunindextype N);

void SUNDenseMatrix_Print(SUNMatrix A, FILE* outfile);

sunindextype SUNDenseMatrix_Rows(SUNMatrix A);
sunindextype SUNDenseMatrix_Columns(SUNMatrix A);
sunindextype SUNDenseMatrix_LData(SUNMatrix A);
realtype* SUNDenseMatrix_Data(SUNMatrix A);
realtype** SUNDenseMatrix_Cols(SUNMatrix A);
realtype* SUNDenseMatrix_Column(SUNMatrix A, sunindextype j);

SUNMatrix_ID SUNMatGetID_Dense(SUNMatrix A);
SUNMatrix SUNMatClone_Dense(SUNMatrix A);
void SUNMatDestroy_Dense(SUNMatrix A);
int SUNMatZero_Dense(SUNMatrix A);
int SUNMatCopy_Dense(SUNMatrix A, SUNMatrix B);
int SUNMatScaleAdd_Dense(realtype c, SUNMatrix A, SUNMatrix B);
int SUNMatScaleAddI_Dense(realtype c, SUNMatrix A);
int SUNMatMatvec_Dense(SUNMatrix A, N_Vector x, N_Vector y);
int SUNMatSpace_Dense(SUNMatrix A, long int *lenrw, long int *leniw);
