import updateAssignments as upasgn
import computeCenters as cc
import ssd as ss


def kmeans(X, initial_centers, iterations, tolerance):
    
    num_data = len(X) 
    
    if num_data == 0:
        return []

    k = len(initial_centers) 
    
    assignments = [0]* num_data
    
    if k >= num_data:
        for i in range(num_data):
            assignments[i] = i
        return assignments

        
    min_dist = 1e100
    
    for iter in range(iterations):
        print "iteration:", iter
        assignments = upasgn.updateAssignments(X, initial_centers) 
        centers = cc.computeCenters(X, assignments, k)
        
        dist = ss.ssd(X, centers, assignments) # objective function
        if min_dist - dist < tolerance or (min_dist - dist)/min_dist < tolerance:

            return centers, assignments
        
        min_dist = dist
    
    return centers, assignments

