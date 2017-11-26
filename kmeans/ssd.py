def ssd(X, centers, assignments):
    
    s = 0 
    n = len(X) 
    for i in range(n):
        c = assignments[i]
        s += sq_Diff(X[i], centers[c])
        
        return s
            #s += sum(pow([[X[x]] for x in i] - [[centers[y]] for y in c], 2)) 
            
        #s += sum(pow([[X[x]] for x in i] - [[centers[y]] for y in c], 2)) 
        
    

def sq_Diff(x, y):
    diff = 0 
    n = len(x) 
    
    for i in range(n):
        diff += (x[i] - y[i]) * (x[i] - y[i]) 
    
    return diff