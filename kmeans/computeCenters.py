import numpy as np

def computeCenters(X, assignments, k):
    
    nDim = len(X[0]) 
    centers = [[0] * nDim for i in range(k)] 

    indx = sorted(set(assignments)) 
    for c in indx:
        
        cluster_index = clusterIndicesComp(c, assignments)
        
        cluster = [X[i] for i in cluster_index]
        if len(cluster) == 0:
            
            centers[c] = [0] * nDim
        else:
            centers[c] = getMean(cluster)
    return centers 


def clusterIndicesComp(p, assignments): 
    return np.array([i for i, x in enumerate(assignments) if x == p])

def getMean(cluster):
    return [float(sum(x)) / len(x) for x in zip(*cluster)]

