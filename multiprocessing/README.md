# Multiprocessing in Python

First, import `Pool` from ```multiprocessing``` module

```python
from multiprocessing import Pool, Process, Value, Array
```

## Basic Usage (Pool)

```python
with Pool(processes=4) as p:
    p.map(f, [1, 2, 3])
    p.map_async(f, [1, 2, 3])
    for i in p.imap(f, [1, 2, 3]):
        print(i)
    for i in p.imap_unordered(f, [1, 2, 3]):
        print(i)
```
This creates a pool with 4 threads running `f` with args `[1, 2, 3]`

There are more methods to handle different situations.

## More Methods

### map_async(f, [1, 2, 3])

- Asynchronous version of `map()` method.

### imap(f, [1, 2, 3])

- Iterable version of `map()` method.

### imap_unordered(f, [1, 2, 3])

- Like `imap()` method but ordering of results is arbitrary.

### starmap(f, [(1, 2), (3, 4)]) / starmap_async

- Used with functions with multiple parameters.

### apply_async(f, (arg1, arg2))

- Run non-blocking `f(arg1, arg2)`

Example:
```python
with Pool(processes=4) as p:
    p.apply_async(f1, (2,))
    p.apply_async(f2, (2,))
```
`f1` and `f2` will run simultaneously.

## Stats

Run `python3 matrix.py < input.txt`

Output (map):
```
addMat - Single: 0.014308452606201172
addMat - Multi:  0.026044130325317383
mulMat - Single: 9.879592180252075
mulMat - Multi:  2.618814468383789
```

Output (map_async):
```
addMat - Single: 0.014676809310913086
addMat - Multi:  0.012178421020507812
mulMat - Single: 9.88506817817688
mulMat - Multi:  0.004485607147216797
```
> The multiprocessing multiplication is ridiculously fast, not sure if this is related to strange magics.

## Basic Usage (Process)

```python
def f(name):
    print('Hello', name)

p = Process(target = f, args = ('Bob',))
p.start()
p.join()
```

This creates a new process running `f('Bob')`, similar to thread programming.

## Shared Memory and Locks

```python
num = Value('d', 0.0)       # d = double
arr = Array('i', range(10)) # i = signed int

with num.get_lock():        # critical section
    num += 1
```

- For typecodes like `d` and `i`, see [this](https://docs.python.org/3/library/array.html#module-array)

- For more complex datatypes, see [this](https://stackoverflow.com/questions/9754034/can-i-create-a-shared-multiarray-or-lists-of-lists-object-in-python-for-multipro)

- Global variables are **inherited**, not shared (can't write back to other process).

## Reference
[Python 3.11.0 Documentation](https://docs.python.org/3/library/multiprocessing.html)

[Python 使用 multiprocessing 模組開發多核心平行運算程式教學與範例](https://officeguide.cc/python-multiprocessing-parallel-computing-tutorial-examples/)