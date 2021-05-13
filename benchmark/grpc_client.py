import benchmark_pb2, benchmark_pb2_grpc

import pickle
import torch
import grpc
import os
import threading
import time
from functools import partial
from statistics import stdev

from concurrent import futures


from common import (
    identity,
    identity_script,
    heavy,
    heavy_script,
    identity_cuda,
    identity_script_cuda,
    heavy_cuda,
    heavy_script_cuda,
    stamp_time,
    compute_delay,
    NUM_RPC,
)


def get_all_results(futs, cuda):
    cpu_tensors = [pickle.loads(f.result().data) for f in futs]
    if cuda:
        cuda_tensors = [t.cuda(0) for t in cpu_tensors]
        return cuda_tensors
    return cpu_tensors


class Client:
    def __init__(self, server_address):
        self.stubs = []
        for _ in range(NUM_RPC):
            channel = grpc.insecure_channel(server_address)
            self.stubs.append(benchmark_pb2_grpc.GRPCBenchmarkStub(channel))



    def measure(self, *, name=None, tensor=None, cuda=False):
        # warmup
        futs = []
        for i in range(NUM_RPC):
            data = pickle.dumps((name, tensor, cuda))
            request = benchmark_pb2.Request(data=data)
            futs.append(self.stubs[i].meta_run.future(request))

        get_all_results(futs, cuda)

        # warmup done
        timestamps = {}

        states = {
            "lock": threading.Lock(),
            "future": futures.Future(),
            "pending": NUM_RPC
        }
        def mark_complete_cpu(index, cuda, fut):
            tensor = pickle.loads(fut.result().data)
            if cuda:
                tensor.cuda(0)
            timestamps[index]["tok"] = stamp_time(cuda)

            with states["lock"]:
                states["pending"] -= 1
                if states["pending"] == 0:
                    states["future"].set_result(0)

        start = time.time()
        futs = []
        for index in range(NUM_RPC):
            timestamps[index] = {}
            timestamps[index]["tik"] = stamp_time(cuda)

            data = pickle.dumps((name, tensor, cuda))
            request = benchmark_pb2.Request(data=data)
            fut = self.stubs[index].meta_run.future(request)
            futs.append(fut)

            fut.add_done_callback(partial(mark_complete_cpu, index, cuda))

        states["future"].result()
        end = time.time()

        delays = []
        for index in range(len(timestamps)):
            delays.append(compute_delay(timestamps[index], cuda))

        mean = sum(delays)/len(delays)
        stdv = stdev(delays)
        print(f"{name}_{'cuda' if cuda else 'cpu'}: mean = {mean}, stdev = {stdv}, total = {end - start}", flush=True)
        return mean, stdv

    def terminate(self):
        self.stubs[0].terminate(benchmark_pb2.EmptyMessage())

def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    assert torch.cuda.device_count() == 1

    client = Client("localhost:29500")

    for size in [100, 1000]:
        print(f"======= size = {size} =====")
        tensor = torch.ones(size, size)
        # identity
        client.measure(
            name="identity",
            tensor=tensor,
            cuda=False
        )

        # identity_script
        client.measure(
            name="identity_script",
            tensor=tensor,
            cuda=False
        )

        # heavy
        client.measure(
            name="heavy",
            tensor=tensor,
            cuda=False,
        )

        # heavy script
        client.measure(
            name="heavy_script",
            tensor=tensor,
            cuda=False,
        )

        tensor = tensor.to(0)
        torch.cuda.current_stream(0).synchronize()
        # identity cuda
        client.measure(
            name="identity",
            tensor=tensor,
            cuda=True,
        )

        # identity_script cuda
        client.measure(
            name="identity_script",
            tensor=tensor,
            cuda=True,
        )

        # heavy cuda
        client.measure(
            name="heavy",
            tensor=tensor,
            cuda=True,
        )

        # heavy_script cuda
        client.measure(
            name="heavy_script",
            tensor=tensor,
            cuda=True,
        )

    client.terminate()
