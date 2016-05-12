##### Define the distance between Jam A and Jam B

import numpy as np
from geopy.distance import vincenty

def DF(A_, B_):
    '''
    The argument A_ and B_ should be two lists
    '''
    A_ = np.array(A_)
    B_ = np.array(B_)
    A = np.reshape(A_, (int(np.size(A_) / 2), 2))
    B = np.reshape(B_, (int(np.size(B_) / 2), 2))
    len_A = np.size(A, 0)
    len_B = np.size(B, 0)

    dist_AB_ = []
    for i in range(len_A):
        for j in range(len_B):
            jam1 = (A[i,1], A[i,0])
            jam2 = (B[j,1], B[j,0])
            dist_AB_.append(vincenty(jam1, jam2).meters)
    if (len(dist_AB_) > 0):
        dist_AB = min(dist_AB_)
    else:
        dist_AB = 1e8
    return dist_AB
