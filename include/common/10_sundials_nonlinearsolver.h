typedef ... *SUNNonlinearSolver;

/* -----------------------------------------------------------------------------
 * Integrator supplied function types
 * ---------------------------------------------------------------------------*/

typedef int (*SUNNonlinSolSysFn)(N_Vector y, N_Vector F, void* mem);

typedef int (*SUNNonlinSolLSetupFn)(N_Vector y, N_Vector F, booleantype jbad,
                                    booleantype* jcur, void* mem);

typedef int (*SUNNonlinSolLSolveFn)(N_Vector y, N_Vector b, void* mem);

typedef int (*SUNNonlinSolConvTestFn)(SUNNonlinearSolver NLS, N_Vector y,
                                      N_Vector del, realtype tol, N_Vector ewt,
                                      void* mem);


/* -----------------------------------------------------------------------------
 * SUNNonlinearSolver types
 * ---------------------------------------------------------------------------*/

typedef enum {
  SUNNONLINEARSOLVER_ROOTFIND,
  SUNNONLINEARSOLVER_FIXEDPOINT
} SUNNonlinearSolver_Type;


/* -----------------------------------------------------------------------------
 * Functions exported by SUNNonlinearSolver module
 * ---------------------------------------------------------------------------*/

/* core functions */
SUNNonlinearSolver_Type SUNNonlinSolGetType(SUNNonlinearSolver NLS);

int SUNNonlinSolInitialize(SUNNonlinearSolver NLS);

int SUNNonlinSolSetup(SUNNonlinearSolver NLS, N_Vector y, void* mem);

int SUNNonlinSolSolve(SUNNonlinearSolver NLS,
                      N_Vector y0, N_Vector y,
                      N_Vector w, realtype tol,
                      booleantype callLSetup, void *mem);

int SUNNonlinSolFree(SUNNonlinearSolver NLS);

/* set functions */
int SUNNonlinSolSetSysFn(SUNNonlinearSolver NLS, SUNNonlinSolSysFn SysFn);
int SUNNonlinSolSetLSetupFn(SUNNonlinearSolver NLS, SUNNonlinSolLSetupFn SetupFn);
int SUNNonlinSolSetLSolveFn(SUNNonlinearSolver NLS, SUNNonlinSolLSolveFn SolveFn);
int SUNNonlinSolSetConvTestFn(SUNNonlinearSolver NLS, SUNNonlinSolConvTestFn CTestFn);
int SUNNonlinSolSetMaxIters(SUNNonlinearSolver NLS, int maxiters);

/* get functions */
int SUNNonlinSolGetNumIters(SUNNonlinearSolver NLS, long int *niters);
int SUNNonlinSolGetCurIter(SUNNonlinearSolver NLS, int *iter);
int SUNNonlinSolGetNumConvFails(SUNNonlinearSolver NLS, long int *nconvfails);


/* -----------------------------------------------------------------------------
 * SUNNonlinearSolver return values
 * ---------------------------------------------------------------------------*/

#define SUN_NLS_SUCCESS        0  /* successful / converged */

/* Recoverable */
#define SUN_NLS_CONTINUE       1  /* not converged, keep iterating      */
#define SUN_NLS_CONV_RECVR     2  /* convergece failure, try to recover */

/* Unrecoverable */
#define SUN_NLS_MEM_NULL      -1  /* memory argument is NULL            */
#define SUN_NLS_MEM_FAIL      -2  /* failed memory access / allocation  */
#define SUN_NLS_ILL_INPUT     -3  /* illegal function input             */
#define SUN_NLS_VECTOROP_ERR  -4  /* failed NVector operation           */
