import numpy as np

def distance(A,B):
    a = A
    b = B
    if(a.shape[0] != b.shape[0]):
        print('A and B should be same dimensionality')
        return
    aa = np.sum(a*a, axis = 0)
    bb = np.sum(b*b, axis = 0)
    return np.sqrt(np.abs(np.reshape(aa,(aa.shape[0],1)) + bb - (2 * np.matmul(a.T,b))))


