#pragma once

#include "tile_helper.h"
#include "utils.h"

namespace standard {
class InIterator {
private:
  using TileLoadA = TileLoad<Type, Size_Block_Row, Depth_Block>;
  using TileLoadB = TileLoad<Type, Size_Block_Col, Depth_Block>;
  TileLoadA tile_a;
  TileLoadB tile_b;
  SliceLoad<Type, Size_Thr_Row, Size_Warp_Row, Size_Block_Row, Depth_Block> slice_a;
  SliceLoad<Type, Size_Thr_Col, Size_Warp_Col, Size_Block_Col, Depth_Block> slice_b;

public:
  struct InSharedMemory {
    typename TileLoadA::shared_memory smem_a;
    typename TileLoadB::shared_memory smem_b;
  };

  MMLT_DEVICE
  InIterator(Array<Type, Size_Thr_Row> *regs_row, Array<Type, Size_Thr_Col> *regs_col,
             Array<Type, Size_Thr_Tot> *regs_tot, InSharedMemory &smem, Params &param,
             idx &id)
      : tile_a((Type *)param.inputs[0], smem.smem_a,
               {param.dim2[0].x, param.dim2[0].y, param.size_K},
               {id.block_m, id.block_k, id.tot_thr}),
        tile_b((Type *)param.inputs[1], smem.smem_b,
               {param.dim2[1].x, param.dim2[1].y, param.size_K},
               {id.block_n, id.block_k, id.tot_thr}),
        slice_a(regs_row, smem.smem_a.tile.storage, {id.warp_m, id.thread_m}),
        slice_b(regs_col, smem.smem_b.tile.storage, {id.warp_n, id.thread_n}) {}

  MMLT_DEVICE
  void reset_smem_offset() {
    slice_a.reset();
    slice_b.reset();
  }

  MMLT_DEVICE
  void load_next_smem() {
    tile_a.load();
    tile_b.load();
    tile_a.next();
    tile_b.next();
  }

  MMLT_DEVICE
  void load_next_reg() {
    slice_a.load();
    slice_b.load();
    slice_a.next();
    slice_b.next();
  }
};

class OutIterator {
private:
  using TileC = TileStoreAgg<Type, atomicAdd>;
  TileLoadVec<int, Size_Thr_Row, Size_Warp_Row, Size_Block_Row> tile_x;
  TileLoadVec<int, Size_Thr_Col, Size_Warp_Col, Size_Block_Col> tile_y;
  TileC tile_c;
  SliceStore<Type> slice_c;

public:
  struct OutSharedMemory {
    typename TileC::shared_memory smem_c;
  };

  MMLT_DEVICE
  OutIterator(Array<Type, Size_Thr_Tot> &out_reg, Array<Type, Size_Thr_Row> *regs_row,
              Array<Type, Size_Thr_Col> *regs_col, Array<Type, Size_Thr_Tot> *regs_tot,
              OutSharedMemory &smem, Params &param, idx &id)
      : tile_x((int *)param.inputs[2], smem.smem_c.ind_x.storage, {param.dim2[2].x}, {id.block_m, id.warp_m, id.thread_m}),
	    tile_y((int *)param.inputs[3], smem.smem_c.ind_y.storage, {param.dim2[3].x}, {id.block_n, id.warp_n, id.thread_n}),
	    tile_c((Type *)param.work, smem.smem_c,
               {param.size_p.kRow, param.size_p.kColumn}, param.dim2[Num_Inputs], id),
        slice_c(smem.smem_c.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg, id) {
  }

  MMLT_DEVICE
  void store_next_smem() {
    tile_c.store();
    tile_c.next();
  }

  MMLT_DEVICE
  void store_next_reg() {
    slice_c.store();
    slice_c.next();
  }
};
} // namespace standard