"""One simulated NPU = one process. Runs the counted pass, then warmup + timed runs."""
import os
import time
import traceback
from collections import defaultdict

import torch

from ..comm.counting import CountingComm
from ..comm.timing import TimingComm
from ..kernels.counting import CountingKernels
from ..kernels.timing import TimingKernels
from ..metrics.tracer import Tracer
from ..registry import COMMS, KERNELS, STRATEGIES
from .generate import run_workload


def pin_threads(rank, threads):
    """torch threads per simulated NPU; also pin to cores where the OS allows it (Linux)."""
    torch.set_num_threads(threads)
    if hasattr(os, "sched_setaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        mine = cores[(rank * threads) % len(cores):][:threads]
        if len(mine) == threads:
            os.sched_setaffinity(0, mine)
            return True
    return False


def worker_main(rank, job, shard):
    out_path = os.path.join(job["out_dir"], f"npu{rank}.pt")
    try:
        torch.set_grad_enabled(False)
        cfg, plan = job["config"], job["plan"]
        pinned = pin_threads(rank, cfg["measurement"]["threads_per_process"])
        comm = COMMS.build(cfg["comm"], rank=rank, world_size=plan["num_npus"],
                           groups=plan["comm_groups"], port=job["port"])
        kernels = KERNELS.build(cfg["kernels"])
        strategy = STRATEGIES.build(cfg["strategy"], fused_qkv=cfg["fused_qkv"], dtype_bytes=job["dtype_bytes"],
                                    **cfg["strategy_args"])
        # Copy the shard out of the inter-process shared memory into this process's own
        # allocations (like an NPU owning its DRAM); BLAS results can depend on alignment.
        shard = {k: v.clone() for k, v in shard.items()}
        strategy.bind(job["spec"], plan, shard)

        # 1) counted run: exact counts and the outputs used for correctness
        tracer = Tracer()
        counted = run_workload(strategy, rank,
                               CountingKernels(kernels, tracer, cfg["dtype"], job["dtype_bytes"]),
                               CountingComm(comm, tracer, job["dtype_bytes"]), tracer, job)

        # 2) laptop timing: warmup + timed runs (wall clock, never used for Reef estimates)
        m = cfg["measurement"]
        timed = []
        for i in range(m["warmup_runs"] + m["timed_runs"]):
            tt = Tracer()
            comm.barrier()
            t0 = time.perf_counter()
            run_workload(strategy, rank, TimingKernels(kernels, tt), TimingComm(comm, tt), tt, job)
            comm.barrier()
            total = time.perf_counter() - t0
            if i >= m["warmup_runs"]:
                layers = defaultdict(lambda: [0.0, 0.0])
                for (_, layer), (k, c) in tt.times.items():
                    layers[layer][0] += k * 1e3
                    layers[layer][1] += c * 1e3
                timed.append({"total_ms": total * 1e3, "layers": dict(layers)})

        torch.save({
            "npu_id": rank, "pinned": pinned,
            "kernel_records": tracer.kernel_records, "comm_records": tracer.comm_records,
            "passes": counted["passes"], "logits": counted["logits"], "tokens": counted["tokens"],
            "kv_cache_elements": counted["kv_cache_elements"], "timed": timed,
        }, out_path)
        comm.barrier()
        comm.close()
    except BaseException:
        with open(out_path + ".err", "w") as f:
            f.write(traceback.format_exc())
        raise
