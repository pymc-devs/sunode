/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds, Ashley Crawford @ SMU
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
 * This is the header file for the dense implementation of the
 * SUNLINSOL module, SUNLINSOL_DENSE.
 *
 * Notes:
 *   - The definition of the generic SUNLinearSolver structure can
 *     be found in the header file sundials_linearsolver.h.
 *   - The definition of the type 'realtype' can be found in the
 *     header file sundials_types.h, and it may be changed (at the
 *     configuration stage) according to the user's needs.
 *     The sundials_types.h file also contains the definition
 *     for the type 'booleantype' and 'indextype'.
 * -----------------------------------------------------------------
 */


/* ----------------------------------------
 * Dense Implementation of SUNLinearSolver
 * ---------------------------------------- */

typedef ... *SUNLinearSolverContent_Dense;

/* ----------------------------------------
 * Exported Functions for SUNLINSOL_DENSE
 * ---------------------------------------- */

SUNLinearSolver SUNLinSol_Dense(N_Vector y, SUNMatrix A);

/* deprecated */
SUNLinearSolver SUNDenseLinearSolver(N_Vector y, SUNMatrix A);

SUNLinearSolver_Type SUNLinSolGetType_Dense(SUNLinearSolver S);
int SUNLinSolInitialize_Dense(SUNLinearSolver S);
int SUNLinSolSetup_Dense(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_Dense(SUNLinearSolver S, SUNMatrix A,
                         N_Vector x, N_Vector b, realtype tol);
long int SUNLinSolLastFlag_Dense(SUNLinearSolver S);
int SUNLinSolSpace_Dense(SUNLinearSolver S,
                         long int *lenrwLS,
                         long int *leniwLS);
int SUNLinSolFree_Dense(SUNLinearSolver S);
