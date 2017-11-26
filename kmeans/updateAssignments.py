import numpy as np
from ssd import *
from scipy.spatial.distance import cdist
import sklearn.metrics.pairwise as pairwise_dist

def updateAssignments(X, centers):
    
    num_data = len(X) 
    num_centers = len(centers) 

    
    assignments = [0]*num_data
    dist = [0]*num_centers
    
    
    for i in range(num_data):
        for j in range(num_centers):
            dist[j] = sq_Diff(X[i], centers[j])
        assignments[i] = dist.index(min(dist))
                    
        #dist = cdist(X[[i]], centers, 'sqeuclidean')
        #dist = pairwise_dist.pairwise_distances(np.array([X[i]]), Y = np.array(centers), metric = "sqeuclidean")
        #assignments[i] = dist.argmin()
    return assignments
    
        
    
