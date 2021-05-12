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
)


NUM_RPC = 2


def get_all_results(futs, cuda):
    cpu_tensors = [pickle.loads(f.result().data) for f in futs]
    if cuda:
        cuda_tensors = [t.cuda(0) for t in cpu_tensors]
        return cuda_tensors
    return cpu_tensors


class Client:
    def __init__(self, server_address):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = benchmark_pb2_grpc.GRPCBenchmarkStub(self.channel)



    def measure(self, *, name=None, func=None, args=None, cuda=False):
        # warmup
        futs = []
        for _ in range(NUM_RPC):
            data = pickle.dumps((func, args))
            request = benchmark_pb2.Request(data=data)
            futs.append(self.stub.meta_run.future(request))

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

            data = pickle.dumps((func, args))
            request = benchmark_pb2.Request(data=data)
            fut = self.stub.meta_run.future(request)
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
        self.stub.terminate(benchmark_pb2.EmptyMessage())

def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    assert torch.cuda.device_count() == 1

    client = Client("localhost:29500")
    tensor = torch.ones(100, 100)
    # identity
    client.measure(
        name="identity",
        func=identity,
        args=(tensor,),
        cuda=False
    )

    client.terminate()
