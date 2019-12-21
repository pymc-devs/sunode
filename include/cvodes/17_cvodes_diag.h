/* ---------------------------------------------------------------------
 * Programmer(s): Radu Serban @ LLNL
 * ---------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ---------------------------------------------------------------------
 * This is the header file for the CVODE diagonal linear solver, CVDIAG.
 * ---------------------------------------------------------------------*/


#define CVDIAG_SUCCESS          0
#define CVDIAG_MEM_NULL        -1
#define CVDIAG_LMEM_NULL       -2
#define CVDIAG_ILL_INPUT       -3
#define CVDIAG_MEM_FAIL        -4

#define CVDIAG_INV_FAIL        -5
#define CVDIAG_RHSFUNC_UNRECVR -6
#define CVDIAG_RHSFUNC_RECVR   -7

/* Return values for adjoint module */

#define CVDIAG_NO_ADJ          -101

/* -----------------
 * Forward Problems
 * ----------------- */

/* CVDiag initialization function */

int CVDiag(void *cvode_mem);

/* Optional output functions */

int CVDiagGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS);
int CVDiagGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS);
int CVDiagGetLastFlag(void *cvode_mem, long int *flag);
char *CVDiagGetReturnFlagName(long int flag);

/* -------------------------------------
 * Backward Problems - Function CVDiagB
 * ------------------------------------- */

int CVDiagB(void *cvode_mem, int which);
