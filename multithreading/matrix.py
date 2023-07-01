import threading
import time

matA = []
matB = []
matC = []
matD = []

def addMat(startRow, totalRow):
    global matA, matB, matC
    for i in range(startRow, startRow + totalRow):
        for j in range(500):
            matC[i][j] = matA[i][j] + matB[i][j]

def mulMat(startRow, totalRow):
    global matA, matB, matD
    for i in range(startRow, startRow + totalRow):
        for j in range(500):
            for k in range(500):
                matD[i][j] += matA[i][k] * matB[k][j]


for i in range(500):
    a = []
    b = []
    for j in range(500):
        a.append(int(input()))
        b.append(0)
    matA.append(a)
    matC.append(b)
    matD.append(b)

for i in range(500):
    a = []
    for j in range(500):
        a.append(int(input()))
    matB.append(a)

start = time.time()

addMat(0, 500)

end = time.time()

print("addMat - Single:", end - start)

start = time.time()

threads = []

for i in range(4):
    threads.append(threading.Thread(target = addMat, args = (125 * i, 125)))
    threads[i].start()
for i in range(4):
    threads[i].join()

end = time.time()

print("addMat - Multi: ", end - start)

start = time.time()

mulMat(0, 500)

end = time.time()

print("mulMat - Single:", end - start)

for i in range(500):
    a = []
    for j in range(500):
        a.append(0)
    matD.append(a)

start = time.time()

threads = []

for i in range(4):
    threads.append(threading.Thread(target = mulMat, args = (125 * i, 125)))
    threads[i].start()
for i in range(4):
    threads[i].join()

end = time.time()

print("mulMat - Multi: ", end - start)