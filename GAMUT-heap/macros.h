/**
 * @file macros.h
 * @author Xincheng Xie (xie.xincheng@columbia.edu)
 * @brief Macros Used in Codes
 * @version 0.1
 * @date 2023-01-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

/**
 * @brief Define the number of inputs. There are four values, total number of
 * inputs, Row-like matrix, Column-like matrix, result-like matrix.
 *
 * Only need to fill corresponding number into the hole.
 *
 * Example:
 *
 * ```C++
 * // Macros
 * #define Num_Inputs 2
 * #define Num_Inputs_Row 1
 * #define Num_Inputs_Col 1
 * #define Num_Inputs_Tot 0
 * ```
 *
 * @return void
 */
/* @0: Input Macros */
#define Num_Inputs 4
#define Num_Inputs_Row 1
#define Num_Inputs_Col 1
#define Num_Inputs_Tot 0

#define Type float

#ifndef MMLT_DEVICE
	#define MMLT_DEVICE __forceinline__ __device__
#endif
#ifndef MMLT_UNROLL
	#define MMLT_UNROLL _Pragma("unroll")
#endif
#ifndef MMLT_LOOP
	#define MMLT_LOOP _Pragma("unroll 1")
#endif
