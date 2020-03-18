/* Default SPGMR solver parameters */
#define SUNSPGMR_MAXL_DEFAULT    5
#define SUNSPGMR_MAXRS_DEFAULT   0
#define SUNSPGMR_GSTYPE_DEFAULT  ...

/* ----------------------------------------
 * SPGMR Implementation of SUNLinearSolver
 * ---------------------------------------- */

struct _SUNLinearSolverContent_SPGMR {
  int maxl;
  int pretype;
  int gstype;
  int max_restarts;
  int numiters;
  realtype resnorm;
  int last_flag;

  ATimesFn ATimes;
  void* ATData;
  PSetupFn Psetup;
  PSolveFn Psolve;
  void* PData;

  N_Vector s1;
  N_Vector s2;
  N_Vector *V;
  realtype **Hes;
  realtype *givens;
  N_Vector xcor;
  realtype *yg;
  N_Vector vtemp;

  realtype *cv;
  N_Vector *Xv;
};

typedef struct _SUNLinearSolverContent_SPGMR *SUNLinearSolverContent_SPGMR;


/* ---------------------------------------
 * Exported Functions for SUNLINSOL_SPGMR
 * --------------------------------------- */

SUNLinearSolver SUNLinSol_SPGMR(N_Vector y,
                                                int pretype,
                                                int maxl);
int SUNLinSol_SPGMRSetPrecType(SUNLinearSolver S,
                                               int pretype);
int SUNLinSol_SPGMRSetGSType(SUNLinearSolver S,
                                             int gstype);
int SUNLinSol_SPGMRSetMaxRestarts(SUNLinearSolver S,
                                                  int maxrs);

SUNLinearSolver_Type SUNLinSolGetType_SPGMR(SUNLinearSolver S);
SUNLinearSolver_ID SUNLinSolGetID_SPGMR(SUNLinearSolver S);
int SUNLinSolInitialize_SPGMR(SUNLinearSolver S);
int SUNLinSolSetATimes_SPGMR(SUNLinearSolver S, void* A_data,
                                             ATimesFn ATimes);
int SUNLinSolSetPreconditioner_SPGMR(SUNLinearSolver S,
                                                     void* P_data,
                                                     PSetupFn Pset,
                                                     PSolveFn Psol);
int SUNLinSolSetScalingVectors_SPGMR(SUNLinearSolver S,
                                                     N_Vector s1,
                                                     N_Vector s2);
int SUNLinSolSetup_SPGMR(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_SPGMR(SUNLinearSolver S, SUNMatrix A,
                                         N_Vector x, N_Vector b, realtype tol);
int SUNLinSolNumIters_SPGMR(SUNLinearSolver S);
realtype SUNLinSolResNorm_SPGMR(SUNLinearSolver S);
N_Vector SUNLinSolResid_SPGMR(SUNLinearSolver S);
sunindextype SUNLinSolLastFlag_SPGMR(SUNLinearSolver S);
int SUNLinSolSpace_SPGMR(SUNLinearSolver S,
                                         long int *lenrwLS,
                                         long int *leniwLS);
int SUNLinSolFree_SPGMR(SUNLinearSolver S);
