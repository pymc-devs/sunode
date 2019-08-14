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
 * This is the header file for the band implementation of the 
 * SUNLINSOL module, SUNLINSOL_BAND.
 *
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 * -----------------------------------------------------------------
 */

/* ---------------------------------------
 * Band Implementation of SUNLinearSolver
 * --------------------------------------- */
  
struct _SUNLinearSolverContent_Band {
  sunindextype  N;
  sunindextype *pivots;
  long int last_flag;
};

typedef struct _SUNLinearSolverContent_Band *SUNLinearSolverContent_Band;

  
/* --------------------------------------
 * Exported Functions for SUNLINSOL_BAND
 * -------------------------------------- */

SUNLinearSolver SUNLinSol_Band(N_Vector y, SUNMatrix A);

SUNLinearSolver_Type SUNLinSolGetType_Band(SUNLinearSolver S);
int SUNLinSolInitialize_Band(SUNLinearSolver S);
int SUNLinSolSetup_Band(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_Band(SUNLinearSolver S, SUNMatrix A,
                        N_Vector x, N_Vector b, realtype tol);
long int SUNLinSolLastFlag_Band(SUNLinearSolver S);
int SUNLinSolSpace_Band(SUNLinearSolver S,
                        long int *lenrwLS,
                        long int *leniwLS);
int SUNLinSolFree_Band(SUNLinearSolver S);
