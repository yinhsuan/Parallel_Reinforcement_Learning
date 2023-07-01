# Threading in Python

First, import ```threading``` module

```python
import threading
```

## Basic Usage

```python
def func(num):
  print("Thread", num)
t = threading.Thread(target = func, args = (1,))
t.start()
t.join()
```

`Thread`: a class

`target`: a callable object specifies which function to run

`args`: a tuple which specifies the args of the function.

Output:
```
Thread 1
```

Use with `for`:

```python
t = []

for i in range(4):
    t.append(threading.Thread(target = job, args = (i,)))
    t[i].start()

for i in range(4):
    t[i].join()
```

Output:
```
Thread 0
Thread 1
Thread 2
Thread 3
```
## Lock / RLock / Semaphore (Same Usage)

```
lock = threading.Lock()
lock.acquire()
lock.release()

rlock = threading.RLock()
rlock.acquire()
rlock.release()

semaphore = threading.Semaphore()
semaphore.acquire()
semaphore.release()
```

Example:
```python
numsum = 0

def job(num):
    global numsum

    lock.acquire()

    print("Thread", num, "acquired the lock")
    numsum += num
    print("numsum =", numsum)
    print("Thread", num, "released the lock")

    lock.release()    

lock = threading.Lock()

t1 = threading.Thread(target = job, args = (1,))
t2 = threading.Thread(target = job, args = (2,))

t1.start()
t2.start()

t1.join()
t2.join()

print("final numsum =", numsum)
```

Notes:

- Lock: One thread **cannot** acquire a lock right after releasing it.

- RLock: One thread **can** acquire a lock right after releasing it.

- Semaphore: A counter initialized to 1.
    
    - Locks when count to 0.
    - +1 when released.
    - -1 when acquired.

## Stats

Run `python3 matrix.py < input.txt` andyou will find that multithreading doesn't improve.

Output:
```
addMat - Single: 0.015233516693115234
addMat - Multi:  0.017266273498535156
mulMat - Single: 11.80352234840393
mulMat - Multi:  12.314005851745605
```

See [this](https://stackoverflow.com/questions/6821477/python-code-performance-decreases-with-threading)

## Reference
[Python 多執行緒 threading 模組平行化程式設計教學](https://blog.gtwang.org/programming/python-threading-multithreaded-programming-tutorial/)