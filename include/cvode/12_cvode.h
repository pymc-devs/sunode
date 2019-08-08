/* -----------------
 * CVODE Constants
 * ----------------- */

/* lmm */
#define CV_ADAMS          1
#define CV_BDF            2

/* itask */
#define CV_NORMAL         1
#define CV_ONE_STEP       2


/* return values */

#define CV_SUCCESS               0
#define CV_TSTOP_RETURN          1
#define CV_ROOT_RETURN           2

#define CV_WARNING              99

#define CV_TOO_MUCH_WORK        -1
#define CV_TOO_MUCH_ACC         -2
#define CV_ERR_FAILURE          -3
#define CV_CONV_FAILURE         -4

#define CV_LINIT_FAIL           -5
#define CV_LSETUP_FAIL          -6
#define CV_LSOLVE_FAIL          -7
#define CV_RHSFUNC_FAIL         -8
#define CV_FIRST_RHSFUNC_ERR    -9
#define CV_REPTD_RHSFUNC_ERR    -10
#define CV_UNREC_RHSFUNC_ERR    -11
#define CV_RTFUNC_FAIL          -12
#define CV_NLS_INIT_FAIL        -13
#define CV_NLS_SETUP_FAIL       -14
#define CV_CONSTR_FAIL          -15

#define CV_MEM_FAIL             -20
#define CV_MEM_NULL             -21
#define CV_ILL_INPUT            -22
#define CV_NO_MALLOC            -23
#define CV_BAD_K                -24
#define CV_BAD_T                -25
#define CV_BAD_DKY              -26
#define CV_TOO_CLOSE            -27
#define CV_VECTOROP_ERR         -28

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------ */

typedef int (*CVRhsFn)(realtype t, N_Vector y,
                       N_Vector ydot, void *user_data);

typedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout,
                        void *user_data);

typedef int (*CVEwtFn)(N_Vector y, N_Vector ewt, void *user_data);

typedef void (*CVErrHandlerFn)(int error_code,
                               const char *module, const char *function,
                               char *msg, void *user_data);

/* -------------------
 * Exported Functions
 * ------------------- */

/* Initialization functions */
void *CVodeCreate(int lmm);

int CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0);
int CVodeReInit(void *cvode_mem, realtype t0, N_Vector y0);

/* Tolerance input functions */
int CVodeSStolerances(void *cvode_mem, realtype reltol, realtype abstol);
int CVodeSVtolerances(void *cvode_mem, realtype reltol, N_Vector abstol);
int CVodeWFtolerances(void *cvode_mem, CVEwtFn efun);

/* Optional input functions */
int CVodeSetErrHandlerFn(void *cvode_mem, CVErrHandlerFn ehfun, void *eh_data);
int CVodeSetErrFile(void *cvode_mem, FILE *errfp);
int CVodeSetUserData(void *cvode_mem, void *user_data);
int CVodeSetMaxOrd(void *cvode_mem, int maxord);
int CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps);
int CVodeSetMaxHnilWarns(void *cvode_mem, int mxhnil);
int CVodeSetStabLimDet(void *cvode_mem, booleantype stldet);
int CVodeSetInitStep(void *cvode_mem, realtype hin);
int CVodeSetMinStep(void *cvode_mem, realtype hmin);
int CVodeSetMaxStep(void *cvode_mem, realtype hmax);
int CVodeSetStopTime(void *cvode_mem, realtype tstop);
int CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef);
int CVodeSetMaxNonlinIters(void *cvode_mem, int maxcor);
int CVodeSetMaxConvFails(void *cvode_mem, int maxncf);
int CVodeSetNonlinConvCoef(void *cvode_mem, realtype nlscoef);
int CVodeSetConstraints(void *cvode_mem, N_Vector constraints);

int CVodeSetNonlinearSolver(void *cvode_mem, SUNNonlinearSolver NLS);

/* Rootfinding initialization function */
int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g);

/* Rootfinding optional input functions */
int CVodeSetRootDirection(void *cvode_mem, int *rootdir);
int CVodeSetNoInactiveRootWarn(void *cvode_mem);

/* Solver function */
int CVode(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask);

/* Dense output function */
int CVodeGetDky(void *cvode_mem, realtype t, int k, N_Vector dky);

/* Optional output functions */
int CVodeGetWorkSpace(void *cvode_mem, long int *lenrw, long int *leniw);
int CVodeGetNumSteps(void *cvode_mem, long int *nsteps);
int CVodeGetNumRhsEvals(void *cvode_mem, long int *nfevals);
int CVodeGetNumLinSolvSetups(void *cvode_mem, long int *nlinsetups);
int CVodeGetNumErrTestFails(void *cvode_mem, long int *netfails);
int CVodeGetLastOrder(void *cvode_mem, int *qlast);
int CVodeGetCurrentOrder(void *cvode_mem, int *qcur);
int CVodeGetNumStabLimOrderReds(void *cvode_mem, long int *nslred);
int CVodeGetActualInitStep(void *cvode_mem, realtype *hinused);
int CVodeGetLastStep(void *cvode_mem, realtype *hlast);
int CVodeGetCurrentStep(void *cvode_mem, realtype *hcur);
int CVodeGetCurrentTime(void *cvode_mem, realtype *tcur);
int CVodeGetTolScaleFactor(void *cvode_mem, realtype *tolsfac);
int CVodeGetErrWeights(void *cvode_mem, N_Vector eweight);
int CVodeGetEstLocalErrors(void *cvode_mem, N_Vector ele);
int CVodeGetNumGEvals(void *cvode_mem, long int *ngevals);
int CVodeGetRootInfo(void *cvode_mem, int *rootsfound);
int CVodeGetIntegratorStats(void *cvode_mem, long int *nsteps,
                            long int *nfevals,
                            long int *nlinsetups,
                            long int *netfails,
                            int *qlast, int *qcur,
                            realtype *hinused, realtype *hlast,
                            realtype *hcur, realtype *tcur);
int CVodeGetNumNonlinSolvIters(void *cvode_mem, long int *nniters);
int CVodeGetNumNonlinSolvConvFails(void *cvode_mem, long int *nncfails);
int CVodeGetNonlinSolvStats(void *cvode_mem, long int *nniters, long int *nncfails);
char *CVodeGetReturnFlagName(long int flag);

/* Free function */
void CVodeFree(void **cvode_mem);
