#pragma once

namespace example1 {
// Inputs Parameters
const int Num_Inputs = 4;
const int Num_Inputs_Row = 2;
const int Num_Inputs_Col = 2;
const int Num_Inputs_Tot = 0;

// Adjustable Parameters
const int Count_Warp_Block_Col = 2;
const int Size_Thr_Col = 8;
const int Block_Depth = 8;

const int Count_Warp_Block_Row = (2 * Count_Warp_Block_Col);
const int Count_Thr_Warp_Row = 4;
const int Size_Thr_Row = Size_Thr_Col;
const int Reduce_Row = 4;
const int Reduce_Col = 32;
const int Reduce_Inner_Iter = 4;

// Fixed Parameters
const int Count_Thr_Warp_Tot = 32;

// Calculated Parameters
const int Count_Thr_Warp_Col = (Count_Thr_Warp_Tot / Count_Thr_Warp_Row);
const int Count_Thr_Block_Row = (Count_Warp_Block_Row * Count_Thr_Warp_Row);
const int Count_Thr_Block_Col = (Count_Warp_Block_Col * Count_Thr_Warp_Col);
const int Count_Thr_Block_Tot = (Count_Thr_Block_Row * Count_Thr_Block_Col);
const int Size_Warp_Row = (Size_Thr_Row * Count_Thr_Warp_Row);
const int Size_Warp_Col = (Size_Thr_Col * Count_Thr_Warp_Col);
const int Size_Block_Row = (Size_Thr_Row * Count_Thr_Block_Row);
const int Size_Block_Col = (Size_Thr_Col * Count_Thr_Block_Col);
const int Size_Thr_Tot = (Size_Thr_Row * Size_Thr_Col);

const int Store_Num_Iter = (Count_Thr_Block_Row * Size_Block_Col / Count_Thr_Block_Tot);
const int Store_Interval = (Count_Thr_Block_Tot / Size_Block_Col);

const int Load_Num_Iter_A = Size_Block_Row * Block_Depth / Count_Thr_Block_Tot;
const int Load_Num_Iter_B = Size_Block_Col * Block_Depth / Count_Thr_Block_Tot;
const int Load_Interval_A = Count_Thr_Block_Tot / Size_Block_Row;
const int Load_Interval_B = Count_Thr_Block_Tot / Size_Block_Col;
}