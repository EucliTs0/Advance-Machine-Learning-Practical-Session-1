import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from sklearn import datasets
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import time

# input: X is normalized features, k is number of principle components
def pca(X, k):
    # take covariance matrix
    cov_matrix = np.cov(X.T)
    # compute eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    # sort by eigenvalues
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    w_matrix = np.array(eig_pairs[0][1])
    for i in xrange(1,k):
        eigenvec = np.array(eig_pairs[i][1])
        w_matrix = np.column_stack((w_matrix, eigenvec))
    return X.dot(w_matrix)

# input: X are features, k is number of principle components,
# kernel is method to use as kernel ("linear", "polynomial", or "rbf"), settings for kernels
def k_pca(X, k, kernel, degree=3, gamma=None):
    if kernel == "linear":
        gram = np.array(pairwise.linear_kernel(X))
    elif kernel == "polynomial":
        gram = np.array(pairwise.polynomial_kernel(X, degree=degree, gamma=gamma))
    else:
        gram = np.array(pairwise.rbf_kernel(X,gamma=gamma))
    N = len(X)
    I_n = np.ones((N,N)) / N            # matrix of 1/N
    # center gram matrix
    gram_centered = gram - I_n.dot(gram) - gram.dot(I_n) + I_n.dot(gram).dot(I_n)
    # solve eigenvalue decomposition of centered gram matrix
    eig_val, eig_vec = np.linalg.eigh(gram_centered)

    # sort by eigenvalues
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # compute normalized eigenvectors
    norm_eigenvecs = []
    for i in xrange(k):
        norm_eigenvecs.append(eig_pairs[i][1] / (sqrt(eig_pairs[i][0])))
    # compute outputs
    output = np.zeros((N,k), dtype=float)
    for n in xrange(N):
        for j in xrange(k):
            for i in xrange(N):
                output[n][j] = output[n][j] + (norm_eigenvecs[j][i] * gram_centered[n][i])
    return output

##### dataset
n_samples = 100
n_features = 20
#dataset = datasets.make_classification(n_samples, n_features, random_state=1)   # classification
#dataset = datasets.make_circles(n_samples)      #circles
#dataset = datasets.make_moons(n_samples)         #moons
dataset = datasets.make_swiss_roll(n_samples)

##### number of components
k = 2
labels = dataset[1]

dataset = np.array(StandardScaler().fit_transform(dataset[0]))

start_time = time.time()
pca1 = pca(dataset, k)
print round(time.time() - start_time, 5)
start_time = time.time()
pca2 = k_pca(dataset, k, "rbf", degree=10, gamma=15)
print round(time.time() - start_time, 5)

plt.figure()
plt.plot(pca2[labels == 1,0], pca2[labels == 1,1],'ro')
plt.plot(pca2[labels == 0,0], pca2[labels == 0,1],'bo')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('PCA - RBF - Gamma 15')
plt.show()

# for swiss roll (no labels)
# plt.figure()
# plt.plot(pca2[:,0], pca2[:,1],'ro')
#
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# plt.title('PCA - RBF - Gamma 15')
# plt.show()

clf = SGDClassifier(loss="log", penalty="l2")
trainX = pca2[0:70]
trainY = labels[0:70]
testX = pca2[70:100]
testY = labels[70:100]
clf.fit(trainX,trainY)
print clf.score(testX,testY)









