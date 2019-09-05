import numpy as np


x = np.arange(0, 9)
l = len(x)

size = 4
count = int(np.ceil(l / size))
xx = np.empty((count,count))

print(len(xx))

for i in range(count):
    print(x[i*size:i*size + size])
    xx[i] = x[i*size:i*size + size]

