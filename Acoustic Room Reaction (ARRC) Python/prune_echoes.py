import numpy as np
from read_lrs_output import call_lrs
from distance import distance

def prune_echoes(images, loudspeaker, minDistance = 2):
    """
    INPUT:
        images - (3,nPoints) - The 3d-locations of the image positions
        loudspeaker - (3,1) - The 3d-location of the loudspeaker
        mindistance - (1,1) - 2
        As == bs == To be used for exps #2 and #3 but can be safely avoided in our case
    OUTPUT:
        A -
        b -
        V -
    """

    imageDistances = np.sqrt(np.sum(np.square(images - np.reshape(loudspeaker,(loudspeaker.shape[0],1))), axis = 0))
    sorted_idx = np.argsort(imageDistances)
    deleted = np.zeros(imageDistances.shape[0])
    A = [[1, 0,  0], [0,  1,  0], [0,  0,  1], [-1,  0,  0,], [0, -1,  0], [0,  0, -1]]
    b = [15, 15, 15, 15, 15, 15]
    for i in np.arange(imageDistances.shape[0]):
        s0 = sorted_idx[i]
        for idx1 in np.arange(i):
            for idx2 in np.arange(i):
                s1 = sorted_idx[idx1]
                s2 = sorted_idx[idx2]
                if((s1 != s2) and (not deleted[s1]) and (not deleted[s2])):
                    p2 = (images[:,s2] + loudspeaker) / 2
                    n2 = images[:,s2] - loudspeaker
                    n2 = n2 / np.linalg.norm(n2)
                    imageSource12 = images[:, s1] + 2 * (np.matmul(np.reshape(p2 - images[:, s1], (1,3)),n2) * n2)
                    if(np.linalg.norm(imageSource12 - images[:, s0]) < minDistance):
                        deleted[s0] = 1
                        print("Discarded IS #:",s0, " (combining)")
        V = call_lrs(A, b)
        if(not deleted[s0]):
            n0 = images[:,s0] - loudspeaker
            n0 = n0/np.linalg.norm(n0)
            p0 = (images[:,s0] + loudspeaker)/2

            A.append(n0.T.tolist())
            b_to_add = np.matmul(n0.T,p0)
            b.append(float(b_to_add))
            IS = np.zeros(len(b))
            IS[len(b)-1] = s0

            V_new = call_lrs(A,b)

            V_array = np.array(V)
            Vnew_array = np.array(V_new)

            D = distance(np.array(Vnew_array[:,1:]).T, np.array(Vnew_array[:,1:]).T)
            D = D + (np.amax(D.flatten()) * np.eye(D.shape[0]))

            if(np.array_equal(V_array,Vnew_array)):
                A = A[:len(A)-1]
                b = b[:len(b)-1]
                IS = IS[:IS.shape[0]-1]
                print('Discarded IS #',s0,'(no intersection)')
            elif(np.amin(D.flatten()) <= minDistance/2):
                A = A[:len(A) - 1]
                b = b[:len(b) - 1]
                IS = IS[:IS.shape[0] - 1]
                print('Discarded IS #', s0, '(vertex proximity)')

    V = call_lrs(A,b)
    return A, b, V
