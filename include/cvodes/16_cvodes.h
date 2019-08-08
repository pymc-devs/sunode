/* -----------------------------------------------------------------
 * Programmer(s): Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the header file for the main CVODES integrator.
 * -----------------------------------------------------------------*/


/* -----------------
 * CVODES Constants
 * ----------------- */

/* lmm */
#define CV_ADAMS          1
#define CV_BDF            2

/* itask */
#define CV_NORMAL         1
#define CV_ONE_STEP       2

/* ism */
#define CV_SIMULTANEOUS   1
#define CV_STAGGERED      2
#define CV_STAGGERED1     3

/* DQtype */
#define CV_CENTERED       1
#define CV_FORWARD        2

/* interp */
#define CV_HERMITE        1
#define CV_POLYNOMIAL     2

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

#define CV_NO_QUAD              -30
#define CV_QRHSFUNC_FAIL        -31
#define CV_FIRST_QRHSFUNC_ERR   -32
#define CV_REPTD_QRHSFUNC_ERR   -33
#define CV_UNREC_QRHSFUNC_ERR   -34

#define CV_NO_SENS              -40
#define CV_SRHSFUNC_FAIL        -41
#define CV_FIRST_SRHSFUNC_ERR   -42
#define CV_REPTD_SRHSFUNC_ERR   -43
#define CV_UNREC_SRHSFUNC_ERR   -44

#define CV_BAD_IS               -45

#define CV_NO_QUADSENS          -50
#define CV_QSRHSFUNC_FAIL       -51
#define CV_FIRST_QSRHSFUNC_ERR  -52
#define CV_REPTD_QSRHSFUNC_ERR  -53
#define CV_UNREC_QSRHSFUNC_ERR  -54

/* adjoint return values */

#define CV_NO_ADJ              -101
#define CV_NO_FWD              -102
#define CV_NO_BCK              -103
#define CV_BAD_TB0             -104
#define CV_REIFWD_FAIL         -105
#define CV_FWD_FAIL            -106
#define CV_GETY_BADT           -107

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

typedef int (*CVQuadRhsFn)(realtype t, N_Vector y,
                           N_Vector yQdot, void *user_data);

typedef int (*CVSensRhsFn)(int Ns, realtype t,
                           N_Vector y, N_Vector ydot,
                           N_Vector *yS, N_Vector *ySdot,
                           void *user_data,
                           N_Vector tmp1, N_Vector tmp2);

typedef int (*CVSensRhs1Fn)(int Ns, realtype t,
                            N_Vector y, N_Vector ydot,
                            int iS, N_Vector yS, N_Vector ySdot,
                            void *user_data,
                            N_Vector tmp1, N_Vector tmp2);

typedef int (*CVQuadSensRhsFn)(int Ns, realtype t,
                               N_Vector y, N_Vector *yS,
                               N_Vector yQdot, N_Vector *yQSdot,
                               void *user_data,
                               N_Vector tmp, N_Vector tmpQ);

typedef int (*CVRhsFnB)(realtype t, N_Vector y, N_Vector yB, N_Vector yBdot,
                        void *user_dataB);

typedef int (*CVRhsFnBS)(realtype t, N_Vector y, N_Vector *yS,
                         N_Vector yB, N_Vector yBdot, void *user_dataB);


typedef int (*CVQuadRhsFnB)(realtype t, N_Vector y, N_Vector yB, N_Vector qBdot,
                            void *user_dataB);

typedef int (*CVQuadRhsFnBS)(realtype t, N_Vector y, N_Vector *yS,
                             N_Vector yB, N_Vector qBdot, void *user_dataB);


/* ---------------------------------------
 * Exported Functions -- Forward Problems
 * --------------------------------------- */

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


/* ---------------------------------
 * Exported Functions -- Quadrature
 * --------------------------------- */

/* Initialization functions */
int CVodeQuadInit(void *cvode_mem, CVQuadRhsFn fQ, N_Vector yQ0);
int CVodeQuadReInit(void *cvode_mem, N_Vector yQ0);

/* Tolerance input functions */
int CVodeQuadSStolerances(void *cvode_mem, realtype reltolQ, realtype abstolQ);
int CVodeQuadSVtolerances(void *cvode_mem, realtype reltolQ, N_Vector abstolQ);

/* Optional input specification functions */
int CVodeSetQuadErrCon(void *cvode_mem, booleantype errconQ);

/* Extraction and Dense Output Functions for Forward Problems */
int CVodeGetQuad(void *cvode_mem, realtype *tret, N_Vector yQout);
int CVodeGetQuadDky(void *cvode_mem, realtype t, int k, N_Vector dky);

/* Optional output specification functions */
int CVodeGetQuadNumRhsEvals(void *cvode_mem, long int *nfQevals);
int CVodeGetQuadNumErrTestFails(void *cvode_mem, long int *nQetfails);
int CVodeGetQuadErrWeights(void *cvode_mem, N_Vector eQweight);
int CVodeGetQuadStats(void *cvode_mem, long int *nfQevals, long int *nQetfails);

/* Free function */
void CVodeQuadFree(void *cvode_mem);


/* ------------------------------------
 * Exported Functions -- Sensitivities
 * ------------------------------------ */

/* Initialization functions */
int CVodeSensInit(void *cvode_mem, int Ns, int ism, CVSensRhsFn fS, N_Vector *yS0);
int CVodeSensInit1(void *cvode_mem, int Ns, int ism, CVSensRhs1Fn fS1, N_Vector *yS0);
int CVodeSensReInit(void *cvode_mem, int ism, N_Vector *yS0);

/* Tolerance input functions */
int CVodeSensSStolerances(void *cvode_mem, realtype reltolS,
                                          realtype *abstolS);
int CVodeSensSVtolerances(void *cvode_mem, realtype reltolS,
                                          N_Vector *abstolS);
int CVodeSensEEtolerances(void *cvode_mem);

/* Optional input specification functions */
int CVodeSetSensDQMethod(void *cvode_mem, int DQtype, realtype DQrhomax);
int CVodeSetSensErrCon(void *cvode_mem, booleantype errconS);
int CVodeSetSensMaxNonlinIters(void *cvode_mem, int maxcorS);
int CVodeSetSensParams(void *cvode_mem, realtype *p, realtype *pbar, int *plist);

/* Integrator nonlinear solver specification functions */
int CVodeSetNonlinearSolverSensSim(void *cvode_mem, SUNNonlinearSolver NLS);
int CVodeSetNonlinearSolverSensStg(void *cvode_mem, SUNNonlinearSolver NLS);
int CVodeSetNonlinearSolverSensStg1(void *cvode_mem, SUNNonlinearSolver NLS);

/* Enable/disable sensitivities */
int CVodeSensToggleOff(void *cvode_mem);

/* Extraction and dense output functions */
int CVodeGetSens(void *cvode_mem, realtype *tret, N_Vector *ySout);
int CVodeGetSens1(void *cvode_mem, realtype *tret, int is, N_Vector ySout);

int CVodeGetSensDky(void *cvode_mem, realtype t, int k, N_Vector *dkyA);
int CVodeGetSensDky1(void *cvode_mem, realtype t, int k, int is, N_Vector dky);

/* Optional output specification functions */
int CVodeGetSensNumRhsEvals(void *cvode_mem, long int *nfSevals);
int CVodeGetNumRhsEvalsSens(void *cvode_mem, long int *nfevalsS);
int CVodeGetSensNumErrTestFails(void *cvode_mem, long int *nSetfails);
int CVodeGetSensNumLinSolvSetups(void *cvode_mem, long int *nlinsetupsS);
int CVodeGetSensErrWeights(void *cvode_mem, N_Vector *eSweight);
int CVodeGetSensStats(void *cvode_mem, long int *nfSevals,
                      long int *nfevalsS, long int *nSetfails,
                      long int *nlinsetupsS);
int CVodeGetSensNumNonlinSolvIters(void *cvode_mem, long int *nSniters);
int CVodeGetSensNumNonlinSolvConvFails(void *cvode_mem, long int *nSncfails);
int CVodeGetStgrSensNumNonlinSolvIters(void *cvode_mem, long int *nSTGR1niters);
int CVodeGetStgrSensNumNonlinSolvConvFails(void *cvode_mem, long int *nSTGR1ncfails);
int CVodeGetSensNonlinSolvStats(void *cvode_mem, long int *nSniters, long int *nSncfails);

/* Free function */
void CVodeSensFree(void *cvode_mem);


/* -------------------------------------------------------
 * Exported Functions -- Sensitivity dependent quadrature
 * ------------------------------------------------------- */

/* Initialization functions */
int CVodeQuadSensInit(void *cvode_mem, CVQuadSensRhsFn fQS, N_Vector *yQS0);
int CVodeQuadSensReInit(void *cvode_mem, N_Vector *yQS0);

/* Tolerance input functions */
int CVodeQuadSensSStolerances(void *cvode_mem, realtype reltolQS, realtype *abstolQS);
int CVodeQuadSensSVtolerances(void *cvode_mem, realtype reltolQS, N_Vector *abstolQS);
int CVodeQuadSensEEtolerances(void *cvode_mem);

/* Optional input specification functions */
int CVodeSetQuadSensErrCon(void *cvode_mem, booleantype errconQS);

/* Extraction and dense output functions */
int CVodeGetQuadSens(void *cvode_mem, realtype *tret, N_Vector *yQSout);
int CVodeGetQuadSens1(void *cvode_mem, realtype *tret, int is, N_Vector yQSout);

int CVodeGetQuadSensDky(void *cvode_mem, realtype t, int k, N_Vector *dkyQS_all);
int CVodeGetQuadSensDky1(void *cvode_mem, realtype t, int k, int is, N_Vector dkyQS);

/* Optional output specification functions */
int CVodeGetQuadSensNumRhsEvals(void *cvode_mem, long int *nfQSevals);
int CVodeGetQuadSensNumErrTestFails(void *cvode_mem, long int *nQSetfails);
int CVodeGetQuadSensErrWeights(void *cvode_mem, N_Vector *eQSweight);
int CVodeGetQuadSensStats(void *cvode_mem, long int *nfQSevals, long int *nQSetfails);

/* Free function */
void CVodeQuadSensFree(void *cvode_mem);


/* ----------------------------------------
 * Exported Functions -- Backward Problems
 * ---------------------------------------- */

/* Initialization functions */

int CVodeAdjInit(void *cvode_mem, long int steps, int interp);

int CVodeAdjReInit(void *cvode_mem);

void CVodeAdjFree(void *cvode_mem);

/* Backward Problem Setup Functions */

int CVodeCreateB(void *cvode_mem, int lmmB, int *which);

int CVodeInitB(void *cvode_mem, int which,
               CVRhsFnB fB,
               realtype tB0, N_Vector yB0);
int CVodeInitBS(void *cvode_mem, int which,
                CVRhsFnBS fBs,
                realtype tB0, N_Vector yB0);
int CVodeReInitB(void *cvode_mem, int which, realtype tB0, N_Vector yB0);

int CVodeSStolerancesB(void *cvode_mem, int which, realtype reltolB, realtype abstolB);
int CVodeSVtolerancesB(void *cvode_mem, int which, realtype reltolB, N_Vector abstolB);

int CVodeQuadInitB(void *cvode_mem, int which, CVQuadRhsFnB fQB, N_Vector yQB0);
int CVodeQuadInitBS(void *cvode_mem, int which, CVQuadRhsFnBS fQBs, N_Vector yQB0);
int CVodeQuadReInitB(void *cvode_mem, int which, N_Vector yQB0);

int CVodeQuadSStolerancesB(void *cvode_mem, int which, realtype reltolQB, realtype abstolQB);
int CVodeQuadSVtolerancesB(void *cvode_mem, int which, realtype reltolQB, N_Vector abstolQB);

/* Solver Function For Forward Problems */

int CVodeF(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask, int *ncheckPtr);


/* Solver Function For Backward Problems */

int CVodeB(void *cvode_mem, realtype tBout, int itaskB);

/* Optional Input Functions For Adjoint Problems */

int CVodeSetAdjNoSensi(void *cvode_mem);

int CVodeSetUserDataB(void *cvode_mem, int which, void *user_dataB);
int CVodeSetMaxOrdB(void *cvode_mem, int which, int maxordB);
int CVodeSetMaxNumStepsB(void *cvode_mem, int which, long int mxstepsB);
int CVodeSetStabLimDetB(void *cvode_mem, int which, booleantype stldetB);
int CVodeSetInitStepB(void *cvode_mem, int which, realtype hinB);
int CVodeSetMinStepB(void *cvode_mem, int which, realtype hminB);
int CVodeSetMaxStepB(void *cvode_mem, int which, realtype hmaxB);
int CVodeSetConstraintsB(void *cvode_mem, int which, N_Vector constraintsB);
int CVodeSetQuadErrConB(void *cvode_mem, int which, booleantype errconQB);

int CVodeSetNonlinearSolverB(void *cvode_mem, int which, SUNNonlinearSolver NLS);

/* Extraction And Dense Output Functions For Backward Problems */

int CVodeGetB(void *cvode_mem, int which, realtype *tBret, N_Vector yB);
int CVodeGetQuadB(void *cvode_mem, int which, realtype *tBret, N_Vector qB);

/* Optional Output Functions For Backward Problems */

void *CVodeGetAdjCVodeBmem(void *cvode_mem, int which);

int CVodeGetAdjY(void *cvode_mem, realtype t, N_Vector y);

typedef struct {
  void *my_addr;
  void *next_addr;
  realtype t0;
  realtype t1;
  long int nstep;
  int order;
  realtype step;
} CVadjCheckPointRec;

int CVodeGetAdjCheckPointsInfo(void *cvode_mem, CVadjCheckPointRec *ckpnt);


/* Undocumented Optional Output Functions For Backward Problems */

/* -----------------------------------------------------------------
 * CVodeGetAdjDataPointHermite
 * -----------------------------------------------------------------
 *    Returns the 2 vectors stored for cubic Hermite interpolation
 *    at the data point 'which'. The user must allocate space for
 *    y and yd. Returns CV_MEM_NULL if cvode_mem is NULL,
 *    CV_ILL_INPUT if the interpolation type previously specified
 *    is not CV_HERMITE, or CV_SUCCESS otherwise.
 * -----------------------------------------------------------------
 * CVodeGetAdjDataPointPolynomial
 * -----------------------------------------------------------------
 *    Returns the vector stored for polynomial interpolation
 *    at the data point 'which'. The user must allocate space for
 *    y. Returns CV_MEM_NULL if cvode_mem is NULL, CV_ILL_INPUT if
 *    the interpolation type previously specified is not
 *    CV_POLYNOMIAL, or CV_SUCCESS otherwise.
 * ----------------------------------------------------------------- */

int CVodeGetAdjDataPointHermite(void *cvode_mem, int which, realtype *t, N_Vector y, N_Vector yd);

int CVodeGetAdjDataPointPolynomial(void *cvode_mem, int which, realtype *t, int *order, N_Vector y);

/* -----------------------------------------------------------------
 * CVodeGetAdjCurrentCheckPoint
 *    Returns the address of the 'active' check point.
 * ----------------------------------------------------------------- */

int CVodeGetAdjCurrentCheckPoint(void *cvode_mem, void **addr);
