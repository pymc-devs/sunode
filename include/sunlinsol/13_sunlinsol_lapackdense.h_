/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
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
 * This is the header file for the LAPACK dense implementation of the 
 * SUNLINSOL module, SUNLINSOL_LINPACKDENSE.
 *
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 * -----------------------------------------------------------------
 */


typedef ... *SUNLinearSolverContent_LapackDense;

  
/* ---------------------------------------------
 * Exported Functions for SUNLINSOL_LAPACKDENSE
 * --------------------------------------------- */

SUNLinearSolver SUNLinSol_LapackDense(N_Vector y, SUNMatrix A);
  
/* deprecated */
SUNLinearSolver SUNLapackDense(N_Vector y, SUNMatrix A);

SUNLinearSolver_Type SUNLinSolGetType_LapackDense(SUNLinearSolver S);
int SUNLinSolInitialize_LapackDense(SUNLinearSolver S);
int SUNLinSolSetup_LapackDense(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_LapackDense(SUNLinearSolver S, SUNMatrix A,
                               N_Vector x, N_Vector b, realtype tol);
long int SUNLinSolLastFlag_LapackDense(SUNLinearSolver S);
int SUNLinSolSpace_LapackDense(SUNLinearSolver S,
                               long int *lenrwLS,
                               long int *leniwLS);
int SUNLinSolFree_LapackDense(SUNLinearSolver S);
