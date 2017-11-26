from sklearn.metrics.pairwise import rbf_kernel

def create_RBFkernel(X, gamma):
    
    K = rbf_kernel(X, gamma = gamma)
    return K


    