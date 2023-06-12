#pragma once

#include "utils.h"

namespace query2 {
MMLT_DEVICE
void mmlt_op(Array<Size_Thr_Row * Size_Thr_Col> &out_reg, Array<Size_Thr_Row> in_reg_row[], Array<Size_Thr_Col> in_reg_col[], Array<Size_Thr_Tot> in_reg_tot[])
{
    Array<Size_Thr_Row> A_reg = in_reg_row[0];
	Array<Size_Thr_Col> B_reg = in_reg_col[0];
	Array<Size_Thr_Col> R_reg = in_reg_col[1];
	MMLT_UNROLL
	for (int i = 0; i < Size_Thr_Row; i++) {
		MMLT_UNROLL
		for (int j = 0; j < Size_Thr_Col; j++) {
			float multi = A_reg.storage[i] * B_reg.storage[j];
			float thres = R_reg.storage[j];
			out_reg.storage[i * Size_Thr_Col + j] += multi + float(multi > thres) * (multi - thres);
		}
	}
}

MMLT_DEVICE
void reduce_op(float &accum, float &work)
{
	accum += work;
}

MMLT_DEVICE
void coda_op(float &accum)
{
	return;
}
}