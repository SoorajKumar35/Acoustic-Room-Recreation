import numpy as np
from math import sqrt, cos, acos

def cubicrootsnp(a, b, c, d):
    nproots = np.roots([a,b,c,d])
    roots = np.zeros(3)
    for i in range(len(nproots)):
        if nproots[i].imag == 0.0:
            roots[i] = nproots[i].real
    return roots

def cubicroots(a, b, c, d):
    '''
    INPUT: a,b,c,d: These are the coeffs of a cubic function
    OUTPUT: roots: A list of roots of the cubic function. Note that complex roots are returned as zero - thus complex conjugate pairs are returned as a pair of zeros.
                   If the complex number has only a real part, then the real part will be returned. All real roots will be returned as is.
    '''

    p = c/a - b*b/a/a/3
    q = (2*b*b*b/a/a/a - 9*b*c/a/a + 27*d/a) / 27
    DD = p*p*p/27 + q*q/4

    temp1 = 0
    temp2 = 0
    y1 = 0
    y2 = 0
    y3 = 0
    u = 0
    v = 0
    phi = 0
    y2i = 0
    y2r = 0

    roots = [-1, -1, -1]

    if(DD < 0):
        phi = acos(-q/2/sqrt(abs(pow(p,3))/27))
        temp1 = 2*sqrt(abs(p)/3)
        y1 =  temp1*cos(phi/3)
        y2 = -temp1*cos((phi+np.pi)/3)
        y3 = -temp1*cos((phi-np.pi)/3)
    else:
        temp1 = -q/2. + sqrt(DD)
        temp2 = -q/2. - sqrt(DD)
        u = pow(abs(temp1),(1./3.))
        v = pow(abs(temp2),(1./3.))
        if (temp1 < 0.):
            u = -u
        if (temp2 < 0.):
            v = -v
        y1  = u + v
        y2r = -(u+v)/2.
        y2i =  (u-v)*sqrt(3.)/2.

    temp1 = b/a/3.

    y1 = y1-temp1

    if(DD < 0.):
        y2 = y2-temp1
        y3 = y3-temp1
    
    else:
        y2r=y2r-temp1
    
    
    if(DD < 0.):
        roots[0] = y1
        roots[1] = y2
        roots[2] = y3
        
    
    elif(DD == 0.):
        roots[0] =  y1
        roots[1] = y2r
        roots[2] = y2r
    
    else:
        roots[0] = y1
        roots[1] = y2r
        roots[2] = y2r
        if (y2i != 0):
            roots[1] = 0
            roots[2] = 0

    return roots

def optimization(D, dim = 3):
    '''
    INPUT: D(numpy array) - the distances between microphones squared and dim - the ambient dimensions       
    OUPUT: dist - the distance ? 
           D_estim - The best EDM matrix based on the optimization
           xy_estim - The xy coords of the wall vertices?
    '''

    iter_max = 150

    n = D.shape[0]

    xy_estim = np.zeros((n,dim))
    xy_init = np.zeros((n,dim))
    d_estim = np.zeros((n,n))

    cost = np.zeros(4)
    cost[0] = 10000


    # NOTE: We assume that the ambient dimension is always three in this implementation. The 2d implementation is simply
    # optimization in two dims rather than 3.

    for iter_count in np.arange(iter_max):
        for sen_ind in np.arange(n):

            # First, we update the x-coords

            a = 4 * n
            b = 0
            c = 0
            d = 0

            for j in np.arange(n):
                b = b + (xy_estim[sen_ind][0] - xy_estim[j][0])
                c = c + 3*(xy_estim[sen_ind][0] - xy_estim[j][0]) * (xy_estim[sen_ind][0] - xy_estim[j][0]) \
                      + (xy_estim[sen_ind][1] - xy_estim[j][1]) * (xy_estim[sen_ind][1] - xy_estim[j][1]) \
                      + (xy_estim[sen_ind][2] - xy_estim[j][2]) * (xy_estim[sen_ind][2] - xy_estim[j][2]) \
                      - D[sen_ind, j]
                d = d + (xy_estim[sen_ind][0] - xy_estim[j][0]) * (
                        (xy_estim[sen_ind][0] - xy_estim[j][0]) * (xy_estim[sen_ind][0] - xy_estim[j][0])
                      + (xy_estim[sen_ind][1] - xy_estim[j][1]) * (xy_estim[sen_ind][1] - xy_estim[j][1])
                      + (xy_estim[sen_ind][2] - xy_estim[j][2]) * (xy_estim[sen_ind][2] - xy_estim[j][2])
                      - D[sen_ind, j])

            b = 12 * b
            c = 4 * c
            d = 4 * d   

            roots = cubicroots(a,b,c,d)

            deltaX_min = roots[0]

            min_cost = cost[0]

            for k in np.arange(1,4):
                cost[k] = 0
                for j in np.arange(n):
                    if(j != sen_ind):
                        cost[k] = cost[k] + ((xy_estim[sen_ind][0] + roots[k-1] - xy_estim[j][0])*(xy_estim[sen_ind][0] + roots[k-1] - xy_estim[j][0])+(xy_estim[sen_ind][1] - xy_estim[j][1])*(xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][2] - xy_estim[j][2])*(xy_estim[sen_ind][2] - xy_estim[j][2]) - D[sen_ind, j]) * ((xy_estim[sen_ind][0] + roots[k-1] - xy_estim[j][0]) * (xy_estim[sen_ind][0] + roots[k-1] - xy_estim[j][0]) + (xy_estim[sen_ind][1] - xy_estim[j][1]) * (xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][2] - xy_estim[j][2])*(xy_estim[sen_ind][2] - xy_estim[j][2]) - D[sen_ind, j])
                if (cost[k] < min_cost):
                    deltaX_min = roots[k-1]
                    min_cost = cost[k]
            xy_estim[sen_ind][0] = xy_estim[sen_ind][0] + deltaX_min

            # Second, we update the y-coords

            a = 4 * n
            b = 0
            c = 0
            d = 0

            for j in np.arange(n):
                b = b + (xy_estim[sen_ind][1] - xy_estim[j][1])
                c = c + 3*(xy_estim[sen_ind][1] - xy_estim[j][1]) * (xy_estim[sen_ind][1] - xy_estim[j][1]) \
                      + (xy_estim[sen_ind][0] - xy_estim[j][0]) * (xy_estim[sen_ind][0] - xy_estim[j][0]) \
                      + (xy_estim[sen_ind][2] - xy_estim[j][2]) * (xy_estim[sen_ind][2] - xy_estim[j][2]) \
                      - D[sen_ind, j]
                d = d + (xy_estim[sen_ind][1] - xy_estim[j][1]) * ((xy_estim[sen_ind][1] - xy_estim[j][1])
                      * (xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][0] - xy_estim[j][0])
                      * (xy_estim[sen_ind][0] - xy_estim[j][0]) + (xy_estim[sen_ind][2] - xy_estim[j][2])
                      * (xy_estim[sen_ind][2] - xy_estim[j][2])
                      - D[sen_ind, j])
            
            b = 12 * b
            c = 4 * c
            d = 4 * d   

            roots = cubicroots(a,b,c,d)

            deltaY_min = roots[0]

            min_cost = cost[0]

            for k in np.arange(1,4):
                cost[k] = 0
                for j in np.arange(n):
                    if (j != sen_ind):
                        cost[k] = cost[k] + ((xy_estim[sen_ind][0] - xy_estim[j][0])*(xy_estim[sen_ind][0] - xy_estim[j][0])+(xy_estim[sen_ind][1]+roots[k-1] - xy_estim[j][1])*(xy_estim[sen_ind][1] +roots[k-1] - xy_estim[j][1]) + (xy_estim[sen_ind][2] - xy_estim[j][2])*(xy_estim[sen_ind][2] - xy_estim[j][2]) - D[sen_ind, j]) * ((xy_estim[sen_ind][0] - xy_estim[j][0])*(xy_estim[sen_ind][0] - xy_estim[j][0])+(xy_estim[sen_ind][1]+roots[k-1] - xy_estim[j][1])*(xy_estim[sen_ind][1] +roots[k-1] - xy_estim[j][1]) + (xy_estim[sen_ind][2] - xy_estim[j][2])*(xy_estim[sen_ind][2] - xy_estim[j][2]) - D[sen_ind, j])
                if(cost[k] < min_cost):
                    deltaY_min = roots[k-1]
                    min_cost = cost[k]
            xy_estim[sen_ind][1] = xy_estim[sen_ind][1] + deltaY_min

            # Finally, we update the z-coords

            a = 4 * n
            b = 0
            c = 0
            d = 0

            for j in np.arange(n):
                b = b + (xy_estim[sen_ind][2] - xy_estim[j][2])
                c = c + 3*(xy_estim[sen_ind][2] - xy_estim[j][2]) * (xy_estim[sen_ind][2] - xy_estim[j][2]) \
                      + (xy_estim[sen_ind][1] - xy_estim[j][1]) * (xy_estim[sen_ind][1] - xy_estim[j][1]) \
                      + (xy_estim[sen_ind][0] - xy_estim[j][0]) * (xy_estim[sen_ind][0] - xy_estim[j][0]) \
                      - D[sen_ind, j]
                d = d + (xy_estim[sen_ind][2] - xy_estim[j][2]) * ((xy_estim[sen_ind][0] - xy_estim[j][0])
                      * (xy_estim[sen_ind][0] - xy_estim[j][0]) + (xy_estim[sen_ind][1] - xy_estim[j][1])
                      * (xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][2] - xy_estim[j][2])
                      * (xy_estim[sen_ind][2] - xy_estim[j][2])
                      - D[sen_ind, j])
             
            b = 12 * b
            c = 4 * c
            d = 4 * d   

            roots = cubicroots(a,b,c,d)

            deltaZ_min = roots[0]

            min_cost = cost[0]

            for k in np.arange(1,4):
                cost[k] = 0
                for j in np.arange(n):
                    if (j != sen_ind):
                        cost[k] = cost[k] + ((xy_estim[sen_ind][0] - xy_estim[j][0])*(xy_estim[sen_ind][0] - xy_estim[j][0])+(xy_estim[sen_ind][1] - xy_estim[j][1])*(xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][2]  + roots[k-1]- xy_estim[j][2])*(xy_estim[sen_ind][2]  + roots[k-1]- xy_estim[j][2])- D[sen_ind, j]) * ((xy_estim[sen_ind][0] - xy_estim[j][0])*(xy_estim[sen_ind][0] - xy_estim[j][0])+(xy_estim[sen_ind][1] - xy_estim[j][1])*(xy_estim[sen_ind][1] - xy_estim[j][1]) + (xy_estim[sen_ind][2]  + roots[k-1]- xy_estim[j][2])*(xy_estim[sen_ind][2]  + roots[k-1]- xy_estim[j][2])- D[sen_ind, j])
                if(cost[k] < min_cost):
                    deltaZ_min = roots[k-1]
                    min_cost = cost[k]
            xy_estim[sen_ind][2] = xy_estim[sen_ind][2] + deltaZ_min

    dist = 0
    for i in np.arange(n):
        for j in np.arange(n):
            dist += ((xy_estim[i][0] - xy_estim[j][0])*(xy_estim[i][0] - xy_estim[j][0]) + (xy_estim[i][1] - xy_estim[j][1])*(xy_estim[i][1] - xy_estim[j][1]) + (xy_estim[i][2] - xy_estim[j][2])*(xy_estim[i][2] - xy_estim[j][2])- D[i,j])*((xy_estim[i][0] - xy_estim[j][0])*(xy_estim[i][0] - xy_estim[j][0]) + (xy_estim[i][1] - xy_estim[j][1])*(xy_estim[i][1] - xy_estim[j][1]) + (xy_estim[i][2] - xy_estim[j][2])*(xy_estim[i][2] - xy_estim[j][2])- D[i,j])
        
    dist = 1./n * sqrt(dist)
        
    for i in np.arange(n):
        for j in np.arange(n):
            d_estim[i][j] = (xy_estim[i][0] - xy_estim[j][0])*(xy_estim[i][0] - xy_estim[j][0]) + (xy_estim[i][1] - xy_estim[j][1])*(xy_estim[i][1] - xy_estim[j][1]) + (xy_estim[i][2] - xy_estim[j][2])*(xy_estim[i][2] - xy_estim[j][2])

    return dist, xy_estim, d_estim
