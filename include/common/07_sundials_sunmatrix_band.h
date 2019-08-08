SUNMatrix SUNBandMatrix(sunindextype N, sunindextype mu, sunindextype ml);

SUNMatrix SUNBandMatrixStorage(sunindextype N, sunindextype mu, sunindextype ml, sunindextype smu);

void SUNBandMatrix_Print(SUNMatrix A, FILE* outfile);

sunindextype SUNBandMatrix_Rows(SUNMatrix A);
sunindextype SUNBandMatrix_Columns(SUNMatrix A);
sunindextype SUNBandMatrix_LowerBandwidth(SUNMatrix A);
sunindextype SUNBandMatrix_UpperBandwidth(SUNMatrix A);
sunindextype SUNBandMatrix_StoredUpperBandwidth(SUNMatrix A);
sunindextype SUNBandMatrix_LDim(SUNMatrix A);
realtype* SUNBandMatrix_Data(SUNMatrix A);
realtype** SUNBandMatrix_Cols(SUNMatrix A);
realtype* SUNBandMatrix_Column(SUNMatrix A, sunindextype j);

SUNMatrix_ID SUNMatGetID_Band(SUNMatrix A);
SUNMatrix SUNMatClone_Band(SUNMatrix A);
void SUNMatDestroy_Band(SUNMatrix A);
int SUNMatZero_Band(SUNMatrix A);
int SUNMatCopy_Band(SUNMatrix A, SUNMatrix B);
int SUNMatScaleAdd_Band(realtype c, SUNMatrix A, SUNMatrix B);
int SUNMatScaleAddI_Band(realtype c, SUNMatrix A);
int SUNMatMatvec_Band(SUNMatrix A, N_Vector x, N_Vector y);
int SUNMatSpace_Band(SUNMatrix A, long int *lenrw, long int *leniw);
