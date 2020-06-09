/* -----------------------------------------------------------------
 * Implemented SUNLinearSolver types and IDs:
 * ----------------------------------------------------------------- */

typedef enum {
  SUNLINEARSOLVER_DIRECT,
  SUNLINEARSOLVER_ITERATIVE,
  SUNLINEARSOLVER_MATRIX_ITERATIVE
} SUNLinearSolver_Type;

typedef enum {
  SUNLINEARSOLVER_BAND,
  SUNLINEARSOLVER_DENSE,
  SUNLINEARSOLVER_KLU,
  SUNLINEARSOLVER_LAPACKBAND,
  SUNLINEARSOLVER_LAPACKDENSE,
  SUNLINEARSOLVER_PCG,
  SUNLINEARSOLVER_SPBCGS,
  SUNLINEARSOLVER_SPFGMR,
  SUNLINEARSOLVER_SPGMR,
  SUNLINEARSOLVER_SPTFQMR,
  SUNLINEARSOLVER_SUPERLUDIST,
  SUNLINEARSOLVER_SUPERLUMT,
  SUNLINEARSOLVER_CUSOLVERSP_BATCHQR,
  SUNLINEARSOLVER_CUSTOM
} SUNLinearSolver_ID;


/* Forward reference for pointer to SUNLinearSolver_Ops object */
typedef struct _generic_SUNLinearSolver_Ops *SUNLinearSolver_Ops;

/* Forward reference for pointer to SUNLinearSolver object */
typedef struct _generic_SUNLinearSolver *SUNLinearSolver;

/* Structure containing function pointers to linear solver operations */
struct _generic_SUNLinearSolver_Ops {
  SUNLinearSolver_Type (*gettype)(SUNLinearSolver);
  SUNLinearSolver_ID   (*getid)(SUNLinearSolver);
  int                  (*setatimes)(SUNLinearSolver, void*, ATimesFn);
  int                  (*setpreconditioner)(SUNLinearSolver, void*,
                                            PSetupFn, PSolveFn);
  int                  (*setscalingvectors)(SUNLinearSolver,
                                            N_Vector, N_Vector);
  int                  (*initialize)(SUNLinearSolver);
  int                  (*setup)(SUNLinearSolver, SUNMatrix);
  int                  (*solve)(SUNLinearSolver, SUNMatrix, N_Vector,
                                N_Vector, realtype);
  int                  (*numiters)(SUNLinearSolver);
  realtype             (*resnorm)(SUNLinearSolver);
  sunindextype         (*lastflag)(SUNLinearSolver);
  int                  (*space)(SUNLinearSolver, long int*, long int*);
  N_Vector             (*resid)(SUNLinearSolver);
  int                  (*free)(SUNLinearSolver);
};

/* A linear solver is a structure with an implementation-dependent
   'content' field, and a pointer to a structure of linear solver
   operations corresponding to that implementation. */
struct _generic_SUNLinearSolver {
  void *content;
  SUNLinearSolver_Ops ops;
};


/* -----------------------------------------------------------------
 * Functions exported by SUNLinearSolver module
 * ----------------------------------------------------------------- */

SUNLinearSolver SUNLinSolNewEmpty();

void SUNLinSolFreeEmpty(SUNLinearSolver S);

SUNLinearSolver_Type SUNLinSolGetType(SUNLinearSolver S);

SUNLinearSolver_ID SUNLinSolGetID(SUNLinearSolver S);

int SUNLinSolSetATimes(SUNLinearSolver S, void* A_data,
                                       ATimesFn ATimes);

int SUNLinSolSetPreconditioner(SUNLinearSolver S, void* P_data,
                                               PSetupFn Pset, PSolveFn Psol);

int SUNLinSolSetScalingVectors(SUNLinearSolver S, N_Vector s1,
                                               N_Vector s2);

int SUNLinSolInitialize(SUNLinearSolver S);

int SUNLinSolSetup(SUNLinearSolver S, SUNMatrix A);

int SUNLinSolSolve(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                                   N_Vector b, realtype tol);

int SUNLinSolNumIters(SUNLinearSolver S);

realtype SUNLinSolResNorm(SUNLinearSolver S);

N_Vector SUNLinSolResid(SUNLinearSolver S);

sunindextype SUNLinSolLastFlag(SUNLinearSolver S);

int SUNLinSolSpace(SUNLinearSolver S, long int *lenrwLS,
                                   long int *leniwLS);

int SUNLinSolFree(SUNLinearSolver S);


/* -----------------------------------------------------------------
 * SUNLinearSolver return values
 * ----------------------------------------------------------------- */

#define SUNLS_SUCCESS               0   /* successful/converged          */

#define SUNLS_MEM_NULL           ...   /* mem argument is NULL          */
#define SUNLS_ILL_INPUT          ...   /* illegal function input        */
#define SUNLS_MEM_FAIL           ...   /* failed memory access          */
#define SUNLS_ATIMES_FAIL_UNREC  ...   /* atimes unrecoverable failure  */
#define SUNLS_PSET_FAIL_UNREC    ...   /* pset unrecoverable failure    */
#define SUNLS_PSOLVE_FAIL_UNREC  ...   /* psolve unrecoverable failure  */
#define SUNLS_PACKAGE_FAIL_UNREC ...   /* external package unrec. fail  */
#define SUNLS_GS_FAIL            ...   /* Gram-Schmidt failure          */
#define SUNLS_QRSOL_FAIL         ...   /* QRsol found singular R        */
#define SUNLS_VECTOROP_ERR       ...   /* vector operation error        */

#define SUNLS_RES_REDUCED         ...  /* nonconv. solve, resid reduced */
#define SUNLS_CONV_FAIL           ...  /* nonconvergent solve           */
#define SUNLS_ATIMES_FAIL_REC     ...  /* atimes failed recoverably     */
#define SUNLS_PSET_FAIL_REC       ...  /* pset failed recoverably       */
#define SUNLS_PSOLVE_FAIL_REC     ...  /* psolve failed recoverably     */
#define SUNLS_PACKAGE_FAIL_REC    ...  /* external package recov. fail  */
#define SUNLS_QRFACT_FAIL         ...  /* QRfact found singular matrix  */
#define SUNLS_LUFACT_FAIL         ...  /* LUfact found singular matrix  */
