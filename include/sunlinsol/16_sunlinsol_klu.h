/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Based on sundials_klu_impl.h and arkode_klu.h/cvode_klu.h/... 
 *     code, written by Carol S. Woodward @ LLNL
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
 * This is the header file for the KLU implementation of the 
 * SUNLINSOL module, SUNLINSOL_KLU.
 * 
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 * -----------------------------------------------------------------
 */


/* Default KLU solver parameters */
#define SUNKLU_ORDERING_DEFAULT  1    /* COLAMD */
#define SUNKLU_REINIT_FULL       1
#define SUNKLU_REINIT_PARTIAL    2

  
/* -------------------------------------
 * Exported Functions for SUNLINSOL_KLU
 * ------------------------------------- */

SUNLinearSolver SUNLinSol_KLU(N_Vector y, SUNMatrix A);
int SUNLinSol_KLUReInit(SUNLinearSolver S, SUNMatrix A,
                        sunindextype nnz, int reinit_type);
int SUNLinSol_KLUSetOrdering(SUNLinearSolver S,
                             int ordering_choice);


SUNLinearSolver_Type SUNLinSolGetType_KLU(SUNLinearSolver S);
int SUNLinSolInitialize_KLU(SUNLinearSolver S);
int SUNLinSolSetup_KLU(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_KLU(SUNLinearSolver S, SUNMatrix A,
                       N_Vector x, N_Vector b, realtype tol);
long int SUNLinSolLastFlag_KLU(SUNLinearSolver S);
int SUNLinSolSpace_KLU(SUNLinearSolver S,
                       long int *lenrwLS,
                       long int *leniwLS);
int SUNLinSolFree_KLU(SUNLinearSolver S);
