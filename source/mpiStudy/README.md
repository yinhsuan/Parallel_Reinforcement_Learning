# mpi4py

## Install

```
$ conda install mpi4py
```

## Run

Command

```
$ mpiexec -n 4 python script.py
```

Options

```
-f {name}                        file containing the host names
-hosts {host list}               comma separated host list
-n/-np {value}                   number of processes
```

## Set up

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
```

## Point-to-Point

### send/recv

```python
# rank 0
comm.send(data, dest=1, tag=11)

# rank 1
data = comm.recv(source=0, tag=11)

# numpy array
comm.Send(data, dest=1, tag=13)
comm.Recv(data, source=0, tag=13)

comm.Send([data, MPI.INT], dest=1, tag=77)
data = numpy.empty(1000, dtype='i')
comm.Recv([data, MPI.INT], source=0, tag=77)
```

### non-blocking

```python
# rank 0
req = comm.isend(data, dest=1, tag=11)
req.wait()

# rank 1
req = comm.irecv(source=0, tag=11)
data = req.wait()

# waitall
req1 = comm.irecv(source=1, tag=11)
req2 = comm.irecv(source=2, tag=11)
req3 = comm.irecv(source=3, tag=11)
waitall([req1, req2, req3], statuses=None)
```

## Collective

```python
data = comm.bcast(data, root=0) # data can even be a dictionary
data = comm.scatter(data, root=0) # len(data) = size
comm.scatterv(sendbuf, recvbuf, root=0) # recvbuf = [data, data_size, data_type]
data = comm.gather(data, root=0)

# numpy array
comm.Bcast(data, root=0)
comm.Scatter(sendbuf, recvbuf, root=0)
comm.Gather(sendbuf, recvbuf, root=0)
comm.Allgather([x,  MPI.DOUBLE], [xg, MPI.DOUBLE]) # len(xg) = len(x) * size
```

## CUDA-aware MPI + Python GPU arrays

- [Installing CUDA-aware MPI](https://kose-y.github.io/blog/2017/12/installing-cuda-aware-mpi/)
- [Notes on parallel/distributed training in PyTorch](https://www.kaggle.com/code/residentmario/notes-on-parallel-distributed-training-in-pytorch)
- [Azure Distributed GPU training guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu)
- [torch.distributed](https://pytorch.org/docs/stable/distributed.html)

## References

- [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html)
- [MPI Scatter, Gather, and Allgather](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)
- [Python并行编程 中文版](https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/index.html)