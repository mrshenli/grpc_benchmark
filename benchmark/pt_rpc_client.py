from common import identity, identity_script, heavy, heavy_script

from torch.distributed import rpc
from functools import partial
from statistics import stdev

import torch
import time
import os

NUM_RPC = 10

def stamp_time(cuda=False):
    if cuda:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(0))
        return event
    else:
        return time.time()


def compute_delay(ts, cuda=False):
    if cuda:
        return ts["tik"].elapsed_time(ts["tok"]) / 1e3
    else:
        return ts["tok"] - ts["tik"]


def wait_all(futs, cuda):
    torch.futures.wait_all(futs)
    if cuda:
        torch.cuda.synchronize(0)


def measure(*, name=None, func=None, args=None, cuda=False):
    # warmup
    futs = []
    for _ in range(NUM_RPC):
        futs.append(rpc.rpc_async("server", func, args=args))

    wait_all(futs, cuda)

    # warmup done
    timestamps = {}
    def mark_complete_cpu(index, cuda, fut):
        timestamps[index]["tok"] = stamp_time(cuda)

    start = time.time()
    futs = []
    for index in range(NUM_RPC):
        timestamps[index] = {}
        timestamps[index]["tik"] = stamp_time(cuda)
        futs.append(
            rpc.rpc_async(
                "server", func, args=args
            ).then(
                partial(mark_complete_cpu, index, cuda)
            )
        )

    wait_all(futs, cuda)
    end = time.time()

    delays = []
    for index in range(len(timestamps)):
        delays.append(compute_delay(timestamps[index], cuda))

    mean = sum(delays)/len(delays)
    stdv = stdev(delays)
    print(f"{name}_{'cuda' if cuda else 'cpu'}: mean = {mean}, stdev = {stdv}, total = {end - start}", flush=True)
    return mean, stdv


def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    assert torch.cuda.device_count() == 1

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads = 256,
            device_maps={"server": {0 : 0}}
    )
    rpc.init_rpc(
        "client",
        rank=1,
        world_size=2,
        rpc_backend_options=options
    )

    for size in [100, 1000]:
        print(f"======= size = {size} =====")
        tensor = torch.ones(size, size)
        # identity
        measure(
            name="identity",
            func=identity,
            args=(tensor,),
            cuda=False
        )

        # identity script
        measure(
            name="identity_script",
            func=identity_script,
            args=(tensor,),
            cuda=False,
        )

        # heavy
        measure(
            name="heavy",
            func=heavy,
            args=(tensor,),
            cuda=False,
        )

        # heavy script
        measure(
            name="heavy_script",
            func=heavy_script,
            args=(tensor,),
            cuda=False,
        )

        tensor = tensor.to(0)
        torch.cuda.current_stream(0).synchronize()
        # identity cuda
        measure(
            name="identity",
            func=identity,
            args=(tensor,),
            cuda=True,
        )

        # identity script cuda
        measure(
            name="identity_script",
            func=identity_script,
            args=(tensor,),
            cuda=True,
        )

        # heavy cuda
        measure(
            name="heavy",
            func=identity,
            args=(tensor,),
            cuda=True,
        )

        # heavy script cuda
        measure(
            name="heavy_script",
            func=heavy_script,
            args=(tensor,),
            cuda=True,
        )

    rpc.shutdown()
