/* ----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *                Scott D. Cohen, Alan C. Hindmarsh and
 *                Radu Serban @ LLNL
 * ----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ----------------------------------------------------------------
 * This is the header file for CVODE's linear solver interface.
 * ----------------------------------------------------------------*/


//#include <sundials/sundials_direct.h>

/*=================================================================
  CVLS Constants
  =================================================================*/

#define CVLS_SUCCESS          0
#define CVLS_MEM_NULL        -1
#define CVLS_LMEM_NULL       -2
#define CVLS_ILL_INPUT       -3
#define CVLS_MEM_FAIL        -4
#define CVLS_PMEM_NULL       -5
#define CVLS_JACFUNC_UNRECVR -6
#define CVLS_JACFUNC_RECVR   -7
#define CVLS_SUNMAT_FAIL     -8
#define CVLS_SUNLS_FAIL      -9


/*=================================================================
  CVLS user-supplied function prototypes
  =================================================================*/

typedef int (*CVLsJacFn)(realtype t, N_Vector y, N_Vector fy,
                         SUNMatrix Jac, void *user_data,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

typedef int (*CVLsPrecSetupFn)(realtype t, N_Vector y, N_Vector fy,
                               booleantype jok, booleantype *jcurPtr,
                               realtype gamma, void *user_data);

typedef int (*CVLsPrecSolveFn)(realtype t, N_Vector y, N_Vector fy,
                               N_Vector r, N_Vector z, realtype gamma,
                               realtype delta, int lr, void *user_data);

typedef int (*CVLsJacTimesSetupFn)(realtype t, N_Vector y,
                                   N_Vector fy, void *user_data);

typedef int (*CVLsJacTimesVecFn)(N_Vector v, N_Vector Jv, realtype t,
                                 N_Vector y, N_Vector fy,
                                 void *user_data, N_Vector tmp);


/*=================================================================
  CVLS Exported functions
  =================================================================*/

int CVodeSetLinearSolver(void *cvode_mem, SUNLinearSolver LS, SUNMatrix A);


/*-----------------------------------------------------------------
  Optional inputs to the CVLS linear solver interface
  -----------------------------------------------------------------*/

int CVodeSetJacFn(void *cvode_mem, CVLsJacFn jac);
int CVodeSetMaxStepsBetweenJac(void *cvode_mem, long int msbj);
int CVodeSetEpsLin(void *cvode_mem, realtype eplifac);
int CVodeSetPreconditioner(void *cvode_mem, CVLsPrecSetupFn pset, CVLsPrecSolveFn psolve);
int CVodeSetJacTimes(void *cvode_mem, CVLsJacTimesSetupFn jtsetup, CVLsJacTimesVecFn jtimes);

/*-----------------------------------------------------------------
  Optional outputs from the CVLS linear solver interface
  -----------------------------------------------------------------*/

int CVodeGetLinWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS);
int CVodeGetNumJacEvals(void *cvode_mem, long int *njevals);
int CVodeGetNumPrecEvals(void *cvode_mem, long int *npevals);
int CVodeGetNumPrecSolves(void *cvode_mem, long int *npsolves);
int CVodeGetNumLinIters(void *cvode_mem, long int *nliters);
int CVodeGetNumLinConvFails(void *cvode_mem, long int *nlcfails);
int CVodeGetNumJTSetupEvals(void *cvode_mem, long int *njtsetups);
int CVodeGetNumJtimesEvals(void *cvode_mem, long int *njvevals);
int CVodeGetNumLinRhsEvals(void *cvode_mem, long int *nfevalsLS);
int CVodeGetLastLinFlag(void *cvode_mem, long int *flag);
char *CVodeGetLinReturnFlagName(long int flag);
