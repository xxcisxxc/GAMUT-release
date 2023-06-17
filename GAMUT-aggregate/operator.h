/**
 * @file operator.h
 * @author Xincheng Xie (xie.xincheng@columbia.edu)
 * @brief All the independent operators
 * @version 0.1
 * @date 2023-01-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include "utils.h"

namespace standard {
/**
 * @brief This is the inner operation performed within each thread. It computes
 * dot product of vector $Size_Thr_Row \times 1$ and vector $1 \times
 * Size_Thr_Col$ in standard matrix multiplication. Despite various inputs in
 * MMLTs, vectors can be categorized into three kinds: $Size_Thr_Row \times 1$,
 * $1 \times Size_Thr_Col$, $Size_Thr_Row \times Size_Thr_Col$.
 *
 * For usage, please see code in `kernel.cu`.
 *
 * There are two holes in this template: Local register definition and inner
 * computation.
 *
 * Example:
 *
 * ```C++
 * // Local Register Definition
 * Array<Type, Size_Thr_Row> A_reg = in_reg_row[0];
 * Array<Type, Size_Thr_Col> B_reg = in_reg_col[0];
 *
 * // Inner Computation
 * out_reg.storage[i * Size_Thr_Col + j] += A_reg.storage[i] * B_reg.storage[j];
 * ```
 *
 * For $C = AB$, we have following parameters,
 *
 * @param out_reg output register
 * @param in_reg_row the array of registers of $A$-operand like operands
 * @param in_reg_col the array of registers of $B$-operand like operands
 * @param in_reg_tot the array of registers of $C$-operand like operands
 * @return void
 */
MMLT_DEVICE
void mmlt_op(Array<Type, Size_Thr_Row * Size_Thr_Col> &out_reg,
             Array<Type, Size_Thr_Row> in_reg_row[],
             Array<Type, Size_Thr_Col> in_reg_col[],
             Array<Type, Size_Thr_Tot> in_reg_tot[]) {
  /* @0: Local Register Definition */
  MMLT_UNROLL
  for (int i = 0; i < Size_Thr_Row; i++) {
    MMLT_UNROLL
    for (int j = 0; j < Size_Thr_Col; j++) {
      /* @1: Inner Computation */
    }
  }
}

MMLT_DEVICE
void reduce_op(Type &accum, Type &work) { accum += work; }

MMLT_DEVICE
void coda_op(Type &accum) { return; }
} // namespace standard