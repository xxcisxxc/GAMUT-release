import numpy as np
import tvm

from tvm.script import tir as T
from tvm import meta_schedule as ms

N = L = M = 16384

import shutil
shutil.rmtree('./tune_tmp')

@tvm.script.ir_module
class MyMatmul:
    @T.prim_func
    def main(A: T.Buffer[(N, L), "float32"],
             B: T.Buffer[(L, M), "float32"],
             C: T.Buffer[(N, M), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(N, L, M):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

target = tvm.target.Target("cuda -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -thread_warp_size=32 -max_threads_per_block=1024 -max_shared_memory_per_block=49152")

ir_database = ms.tune_tir(
    mod=MyMatmul,
    target=target,
    max_trials_global=100,
    num_trials_per_iter=100,
    work_dir="./tune_tmp",
    task_name="main"
)


shed = ms.tir_integration.compile_tir(ir_database, MyMatmul, target)

func = tvm.build(shed.mod, target=target)

a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)

dev = tvm.cuda()
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
c_tvm = tvm.nd.empty((N, M), device=dev)

func(a_tvm, b_tvm, c_tvm)

evaluator = func.time_evaluator(func.entry_name, dev, number=3)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
)


