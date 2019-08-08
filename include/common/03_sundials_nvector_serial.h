N_Vector N_VNew_Serial(sunindextype vec_length);
N_Vector N_VNewEmpty_Serial(sunindextype vec_length);
N_Vector N_VMake_Serial(sunindextype vec_length, realtype *v_data);

N_Vector *N_VCloneVectorArray_Serial(int count, N_Vector w);

N_Vector *N_VCloneVectorArrayEmpty_Serial(int count, N_Vector w);

void N_VDestroyVectorArray_Serial(N_Vector *vs, int count);

sunindextype N_VGetLength_Serial(N_Vector v);

void N_VPrint_Serial(N_Vector v);

void N_VPrintFile_Serial(N_Vector v, FILE *outfile);

N_Vector_ID N_VGetVectorID_Serial(N_Vector v);
N_Vector N_VCloneEmpty_Serial(N_Vector w);
N_Vector N_VClone_Serial(N_Vector w);
void N_VDestroy_Serial(N_Vector v);
void N_VSpace_Serial(N_Vector v, sunindextype *lrw, sunindextype *liw);
realtype *N_VGetArrayPointer_Serial(N_Vector v);
void N_VSetArrayPointer_Serial(realtype *v_data, N_Vector v);
