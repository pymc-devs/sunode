typedef ... *SUNMatrix;

typedef enum {
  SUNMATRIX_DENSE,
  SUNMATRIX_BAND,
  SUNMATRIX_SPARSE,
  SUNMATRIX_CUSTOM
} SUNMatrix_ID;

SUNMatrix_ID SUNMatGetID(SUNMatrix A);
SUNMatrix SUNMatClone(SUNMatrix A);
void SUNMatDestroy(SUNMatrix A);
int SUNMatZero(SUNMatrix A);
int SUNMatCopy(SUNMatrix A, SUNMatrix B);
int SUNMatScaleAdd(realtype c, SUNMatrix A, SUNMatrix B);
int SUNMatScaleAddI(realtype c, SUNMatrix A);
int SUNMatMatvec(SUNMatrix A, N_Vector x, N_Vector y);
int SUNMatSpace(SUNMatrix A, long int *lenrw, long int *leniw);
