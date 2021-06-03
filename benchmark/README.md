## Run Cross-Machine PyTorch RPC Experiments

Run the following two commands on the caller and callee respectively. Configure the `IFNAME` and `--master_addr` based on your local environment. 

```
GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=client --comm=ptrpc --master_addr=learnfair100
GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=server --comm=ptrpc --master_addr=learnfair100
```

## Run Cross-Machine gRPC Experiments

Run the following two commands on the caller and callee respectively. Configure the `--master_addr` based on your local environment. 

```
python multi_machine_launch.py --role=client --comm=grpc --master_addr=learnfair100
python multi_machine_launch.py --role=server --comm=grpc --master_addr=learnfair100
```

## Run Single-Machine Experiments

```
python single_machine_launch.py
```
