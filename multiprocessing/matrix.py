from multiprocessing import Pool, Value, Array
import time

mat0 = [[0 for x in range(500)] for y in range(500)] 
matA = [[0 for x in range(500)] for y in range(500)]
matB = [[0 for x in range(500)] for y in range(500)]
matC = mat0
matD = mat0
matE = mat0
matF = mat0

def addMat(row):
    global matA, matB
    for j in range(500):
        matC[row][j] = matA[row][j] + matB[row][j]

def mulMat(row):
    global matA, matB
    for j in range(500):
        for k in range(500):
            matD[row][j] += matA[row][k] * matB[row][j]

# initialize
for i in range(500):
    for j in range(500):
        matA[i][j] = int(input())
for i in range(500):
    for j in range(500):
        matB[i][j] = int(input())



#calculate matrix addition singleprocessing
start = time.time()
for i in range(500):
    addMat(i)
end = time.time()
print("addMat - Single:", end - start)

# calculate matrix addition multiprocessing
matE = matC
for i in range(500):
    for j in range(500):
        matC[i][j] = 0
start = time.time()
with Pool(processes = 4) as p:
    p.map_async(addMat, range(500))
end = time.time()
if (matE == matC):
    print("addMat - Multi: ", end - start)

# calculate matrix multiplication singleprocessing
start = time.time()
for i in range(500):
    mulMat(i)
end = time.time()
print("mulMat - Single:", end - start)

# calculate matrix multiplication multiprocessing
matF = matD
for i in range(500):
    for j in range(500):
        matD[i][j] = 0
start = time.time()
with Pool(processes = 4) as p:
    p.map_async(mulMat, range(500))
end = time.time()
if (matF == matD):
    print("mulMat - Multi: ", end - start)
