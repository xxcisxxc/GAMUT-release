/**
 * @file iterator.h
 * @author Xincheng Xie (xie.xincheng@columbia.edu)
 * @brief Tile Transformation Model
 * @version 0.1
 * @date 2023-01-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "tile_helper.h"
#include "utils.h"
/* @12: Custom Libraries including */

namespace standard {
/**
 * @brief This class is an input iterator. Within this big one, there are small
 * iterators for each operand. Usually, the input interator consists of two
 * steps: from global memory (gmem) to shared memory (smem), and smem to
 * register (reg). Refer to `kernel.cu` for concrete usage.
 *
 */
class InIterator {
private:
  /**
   * @brief Declaration of iterators. `*LoadRow` and `*LoadCol` are pre-defined
   * because it is commonly used in $C = AB$. `TileLoad*` is for gmem->smem and
   * `SliceLoad*` is for smem->reg.
   *
   * The hole here is used for declaration.
   *
   * Example:
   *
   * ```C++
   * // Iterator Declaration
   * TileLoadRow tile_a;
   * TileLoadCol tile_b;
   * SliceLoadRow slice_a;
   * SliceLoadCol slice_b;
   * ```
   */
  using TileLoadRow = TileLoad<Type, Size_Block_Row, Depth_Block>;
  using TileLoadCol = TileLoad<Type, Size_Block_Col, Depth_Block>;
  using SliceLoadRow =
      SliceLoad<Type, Size_Thr_Row, Size_Warp_Row, Size_Block_Row, Depth_Block>;
  using SliceLoadCol =
      SliceLoad<Type, Size_Thr_Col, Size_Warp_Col, Size_Block_Col, Depth_Block>;
  /* @0: Iterator Declaration */

public:
  /**
   * @brief Define the structure of shared memory. Shared memory of each small
   * iterators is already defined inside their classes.
   *
   * The hole here is used to apply all the small iterators' smem.
   *
   * Example:
   *
   * ```C++
   * // Shared Memory Definition
   * typename TileLoadRow::shared_memory smem_a;
   * typename TileLoadCol::shared_memory smem_b;
   * ```
   */
  struct InSharedMemory {
    /* @1: Shared Memory Definition */
  };

  /**
   * @brief Constructor that initializes iterators.
   *
   * The hole here is used for initialization.
   *
   * Example:
   *
   * ```C++
   * // Iterator Initialization
   * tile_a((Type *)param.inputs[0], smem.smem_a, {param.dim2[0].x,
   * param.dim2[0].y, param.size_K}, {id.block_m, id.block_k, id.tot_thr}),
   * tile_b((Type *)param.inputs[1], smem.smem_b, {param.dim2[1].x,
   * param.dim2[1].y, param.size_K}, {id.block_n, id.block_k, id.tot_thr}),
   * slice_a(regs_row, smem.smem_a.tile.storage, {id.warp_m, id.thread_m}),
   * slice_b(regs_col, smem.smem_b.tile.storage, {id.warp_n, id.thread_n})
   * ```
   *
   * @param regs_row
   * @param regs_col
   * @param regs_tot
   * @param smem
   * @param param
   * @param id
   * @return void
   */
  MMLT_DEVICE
  InIterator(Array<Type, Size_Thr_Row> *regs_row,
             Array<Type, Size_Thr_Col> *regs_col,
             Array<Type, Size_Thr_Tot> *regs_tot, InSharedMemory &smem,
             Params &param, idx &id)
      : /* @2: Iterator Initialization */
  {}

  /**
   * @brief Reset pointers in slice iterator to the beginning of smem. For
   * concrete usage, refer to `kernel.cu`.
   *
   * Only need to fill corresponding call into the hole.
   *
   * Example:
   *
   * ```C++
   * // Slice Reset
   * slice_a.reset();
   * slice_b.reset();
   * ```
   *
   * @return void
   */
  MMLT_DEVICE
  void reset_smem_offset() {
    /* @3: Slice Reset */
  }

  /**
   * @brief Load a tile from gmem to smem and point to the next position in
   * gmem. For concrete usage, refer to `kernel.cu`.
   *
   * Only need to fill corresponding call into the hole.
   *
   * Example:
   *
   * ```C++
   * // Tile Load
   * tile_a.load();
   * tile_b.load();
   * tile_a.next();
   * tile_b.next();
   * ```
   *
   * @return void
   */
  MMLT_DEVICE
  void load_next_smem() {
    /* @4: Tile Load */
  }

  /**
   * @brief Load a slice from smem to reg and point to the next position in
   * smem. For concrete usage, refer to `kernel.cu`.
   *
   * Only need to fill corresponding call into the hole.
   *
   * Example:
   *
   * ```C++
   * // Slice Load
   * slice_a.load();
   * slice_b.load();
   * slice_a.next();
   * slice_b.next();
   * ```
   *
   * @return void
   */
  MMLT_DEVICE
  void load_next_reg() {
    /* @5: Slice Load */
  }
};

/**
 * @brief the comparison operation that passed into the template of
 * selection/unlinear.
 *
 * The hole is a function.
 *
 * Example:
 *
 * ```C++
 * MMLT_DEVICE
 * bool lessThan(Type num)
 * {
 *    return num < 10000;
 * }
 *
 */
/* @6: Comparator */

/**
 * @brief Output $8\times 8$ register into global memory. Because of limited
 * size of smem, we output $1\times 8$ reg each iteration.  Refer to `kernel.cu`
 * for concrete usage.
 *
 */
class OutIterator {
private:
  /**
   * @brief Declaration of iterators. Mainly two kinds: TileStore and
   * SliceStore.
   *
   * The hole here is used for declaration.
   *
   * Example:
   *
   * ```C++
   * // Iterator Declaration
   * using Tile = TileStore<Type>;
   * Tile tile;
   * SliceStore<Type> slice;
   * ```
   */
  /* @7: Iterator Declaration */

public:
  /**
   * @brief Define the structure of shared memory. Shared memory of each small
   * iterators is already defined inside their classes.
   *
   * The hole here is used to apply all the small iterators' smem.
   *
   * Example:
   *
   * ```C++
   * // Shared Memory Definition
   * typename Tile::shared_memory smem;
   * ```
   */
  struct OutSharedMemory {
    /* @8: Shared Memory Definition */
  };

  /**
   * @brief Constructor that initializes iterators.
   *
   * The hole here is used for initialization.
   *
   * Example:
   *
   * ```C++
   * // Iterator Initialization
   * tile((Type *)param.work, smem.smem, param.dim2[Num_Inputs], id),
   * slice(smem.smem.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg,
   * id)
   * ```
   *
   * @param out_reg
   * @param regs_row
   * @param regs_col
   * @param regs_tot
   * @param smem
   * @param param
   * @param id
   * @return void
   */
  MMLT_DEVICE
  OutIterator(Array<Type, Size_Thr_Tot> &out_reg,
              Array<Type, Size_Thr_Row> *regs_row,
              Array<Type, Size_Thr_Col> *regs_col,
              Array<Type, Size_Thr_Tot> *regs_tot, OutSharedMemory &smem,
              Params &param, idx &id)
      : /* @9: Iterator Initialization */
  {}

  /**
   * @brief Store a tile from smem to gmem and point to the next position in
   * smem. For concrete usage, refer to `kernel.cu`.
   *
   * Only need to fill corresponding call into the hole.
   *
   * Example:
   *
   * ```C++
   * // Tile Store
   * tile.store();
   * tile.next();
   * ```
   *
   * @return void
   */
  MMLT_DEVICE
  void store_next_smem() {
    /* @10: Tile Store */
  }

  /**
   * @brief Store a slice from reg to smem and point to the next position in
   * reg. For concrete usage, refer to `kernel.cu`.
   *
   * Only need to fill corresponding call into the hole.
   *
   * Example:
   *
   * ```C++
   * // Slice Store
   * slice.store();
   * slice.next();
   * ```
   *
   * @return void
   */
  MMLT_DEVICE
  void store_next_reg() {
    /* @11: Slice Store */
  }
};
} // namespace standard