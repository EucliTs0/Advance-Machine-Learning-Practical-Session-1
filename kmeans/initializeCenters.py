#Initialize centers
import numpy as np

def initializeCenters(X, k):
    centers = np.zeros((k, X.shape[1]))
    indx = np.random.permutation(X.shape[0])
    centers = X[indx[:k], :]
    
    return centers

