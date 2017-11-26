#demo kmeans

import numpy as np
from sklearn import datasets
import initializeCenters as ic
from k_means import *
import create_RBFkernel as cRBF
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel
import time

np.random.seed(100)
n_samples = 1000

#X = datasets.make_circles(n_samples)      #circles
X = datasets.make_moons(n_samples)         #moons
X = X[0]
KK = cRBF.create_RBFkernel(X, 15)
#KK = polynomial_kernel(X, degree=8)

k = 2
initial_centers = ic.initializeCenters(X, k)
initial_centers = initial_centers.tolist()

initial_centers_kernel = ic.initializeCenters(KK, k)
initial_centers_kernel = initial_centers_kernel.tolist()

X1 = X.tolist()
KK1 = KK.tolist()

iterations = 100
tolerance = 10e-6
start_time = time.time()
centers, labels = kmeans(X1, initial_centers, iterations, tolerance)
print round(time.time() - start_time, 5)
start_time = time.time()
centers2, labels2 = kmeans(KK1, initial_centers_kernel, iterations, tolerance)
print round(time.time() - start_time, 5)

labels = np.array(labels)
labels2 = np.array(labels2)

plt.figure()
plt.plot(X[labels2 == 0,0], X[labels2 == 0,1],'bo')
plt.plot(X[labels2 == 1,0], X[labels2 == 1,1],'ro')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Polynomial K-Means (degree=8)')
plt.show()