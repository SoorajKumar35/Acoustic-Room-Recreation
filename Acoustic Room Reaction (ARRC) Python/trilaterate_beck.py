import numpy as np
import scipy.linalg as la

def trilaterate_back(microphone_locs, dists_to_loudspeaker):
	'''
	INPUT: microphone_locs = (5,3) np.array - locs of each of the 5 mics in 3d space
		   dists_to_loudspeaker = (5,1) np.array - dists from each mic to loudspeaker
	OUTPUT: estimated_location = (3,1) np.array - loc of loudspeaker
	        total_error = (1,1) total error from estimated location of loudspeaker and true location
	'''


	d = microphone_locs.shape[1]
	m = microphone_locs.shape[0]

	A = -2 * microphone_locs
	ones_to_append = np.ones((m,1))
	A = np.append(A, ones_to_append, axis = 1) # (5,4)

	b = np.square(dists_to_loudspeaker) - np.sum(np.square(microphone_locs), axis = 1)
	b = np.reshape(b,(5,1))

	D = np.append(np.eye(d), np.zeros((d,1)),axis = 1)
	D = np.append(D, np.zeros((1, d+1)), axis = 0) # (4,4)

	f = np.zeros((d+1,1)) # (4,1)
	f[3][0] = -0.5

	y = lambda x: la.solve((np.matmul(A.T,A) + (x*D)),(np.matmul(A.T,b) - (x * f)))
	phi = lambda x: np.matmul(np.matmul(y(x).T,D),y(x)) + (2 * np.matmul(f.T,y(x)))

	np.savetxt('A_matrix.txt', np.matmul(A.T,A))
	np.savetxt('D_matrix.txt', D)

	input_mat = np.matmul(A.T,A)
	eig_vals_vecs = la.eig(D,input_mat)
	eigDAA = eig_vals_vecs[0]
	eigDAA = np.sort(np.reshape(eigDAA, (D.shape[0],)))
	lambda1 = eigDAA[3]

	a1 = -1 / lambda1
	a2 = 1000

	epsAbs = 1e-5
	epsStep = 1e-5

	c = 0

	while ( (a2 - a1 >= epsStep) or  ( abs( phi(a1) ) >= epsAbs) and ( abs( phi(a2) )  >= epsAbs ) ):
		c = (a1 + a2)/2
		if ( phi(c) == 0 ):
			break
		elif ( phi(a1)*phi(c) < 0 ):
			a2 = c
		else:
			a1 = c

	output = y(c)
	estimated_location = output[0:3]

	total_error = np.sum(np.abs(np.sqrt(np.sum(np.square(microphone_locs.T - np.real(estimated_location)),axis = 0)) - dists_to_loudspeaker))

	return np.reshape(estimated_location,(estimated_location.shape[0],)), total_error
