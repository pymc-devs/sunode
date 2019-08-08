typedef enum {
  SUNDIALS_NVEC_SERIAL,
  SUNDIALS_NVEC_PARALLEL,
  SUNDIALS_NVEC_OPENMP,
  SUNDIALS_NVEC_PTHREADS,
  SUNDIALS_NVEC_PARHYP,
  SUNDIALS_NVEC_PETSC,
  SUNDIALS_NVEC_CUDA,
  SUNDIALS_NVEC_RAJA,
  SUNDIALS_NVEC_OPENMPDEV,
  SUNDIALS_NVEC_TRILINOS,
  SUNDIALS_NVEC_CUSTOM
} N_Vector_ID;


N_Vector_ID N_VGetVectorID(N_Vector w);
N_Vector N_VClone(N_Vector w);
N_Vector N_VCloneEmpty(N_Vector w);
void N_VDestroy(N_Vector v);
void N_VSpace(N_Vector v, sunindextype *lrw, sunindextype *liw);
realtype *N_VGetArrayPointer(N_Vector v);
void N_VSetArrayPointer(realtype *v_data, N_Vector v);
void N_VLinearSum(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z);
void N_VConst(realtype c, N_Vector z);
void N_VProd(N_Vector x, N_Vector y, N_Vector z);
void N_VDiv(N_Vector x, N_Vector y, N_Vector z);
void N_VScale(realtype c, N_Vector x, N_Vector z);
void N_VAbs(N_Vector x, N_Vector z);
void N_VInv(N_Vector x, N_Vector z);
void N_VAddConst(N_Vector x, realtype b, N_Vector z);
realtype N_VDotProd(N_Vector x, N_Vector y);
realtype N_VMaxNorm(N_Vector x);
realtype N_VWrmsNorm(N_Vector x, N_Vector w);
realtype N_VWrmsNormMask(N_Vector x, N_Vector w, N_Vector id);
realtype N_VMin(N_Vector x);
realtype N_VWL2Norm(N_Vector x, N_Vector w);
realtype N_VL1Norm(N_Vector x);
void N_VCompare(realtype c, N_Vector x, N_Vector z);
booleantype N_VInvTest(N_Vector x, N_Vector z);
booleantype N_VConstrMask(N_Vector c, N_Vector x, N_Vector m);
realtype N_VMinQuotient(N_Vector num, N_Vector denom);
N_Vector *N_VCloneEmptyVectorArray(int count, N_Vector w);
N_Vector *N_VCloneVectorArray(int count, N_Vector w);
void N_VDestroyVectorArray(N_Vector *vs, int count);
