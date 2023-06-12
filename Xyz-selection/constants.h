/**
 * @file constants.h
 * @author Xincheng Xie (xie.xincheng@columbia.edu)
 * @brief Define Configuration Numbers
 * @version 0.1
 * @date 2023-01-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

namespace standard {
// Adjustable Parameters
const int Count_Warp_Block_Col = 2;
const int Size_Thr_Col = 8;
const int Depth_Block = 8;

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
} // namespace standard