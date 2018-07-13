#! /usr/bin/env python3

from dist_opt_mex_3d import optimization
from time import time
import numpy as np
from trilaterate_beck import trilaterate_back
from prune_echoes import prune_echoes
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings

# toplevel: roomshape
# input  - int: (1-3) Number of Dokmanic's test data to run
# output - list(xyz): List of the corners of the room
#          list(xyz): List of the microphone positions
#          (xyz):     Speaker position
def roomshape(exp_number):
    if exp_number is 1:
        D = np.matrix(
                [[00.0, 57.0, 43.0, 51.6, 21.0],
                 [00.0, 00.0, 25.0, 43.5, 44.0],
                 [00.0, 00.0, 00.0, 29.3, 25.0],
                 [00.0, 00.0, 00.0, 00.0, 32.3],
                 [00.0, 00.0, 00.0, 00.0, 00.0]])
        D = D/100
        D = D + D.transpose()
        samp_direct = [1102, 1168, 1111, 1048, 1083]
        samp_echoes = [[1343, 1479, 1643, 1798, 1927, 2070, 2131, 2249, 2616, 2734, 2958, 3038, 3139, 3212, 3390, 3495, 3974],
                       [1403, 1510, 1683, 1844, 2069, 2212, 2240, 2370, 2545, 2808, 2898, 3106, 3155, 3292, 3355, 3438, 3618, 3970],
                       [1376, 1451, 1632, 1808, 2006, 2161, 2227, 2317, 2358, 2653, 2709, 2847, 3095, 3196, 3284, 3386, 3998],
                       [1299, 1445, 1566, 1730, 1975, 2116, 2279, 2376, 2748, 2834, 2916, 3027, 3144, 3337, 3496, 3868, 3913],
                       [1338, 1454, 1616, 1781, 1948, 2097, 2174, 2299, 2377, 2648, 2740, 2857, 2907, 3109, 3178, 3350, 3445, 3889, 3948]]
        return peaks_to_room_shape(D, samp_direct, samp_echoes, 338)
    else:
        return [], [], []

# peaks_to_room_shape
# input  - matrix: N x N matrix of microphone distances (unsquared)
#          list(sample): size N list of the sample index of the original impulse
#          list(list(sample)): size N list, each element is list of echoes per mic
# output - list(xyz): List of the corners of the room
#          list(xyz): List of the microphone positions
#          (xyz):     Speaker position
def peaks_to_room_shape(D, samp_direct, samp_echoes, delay):
    direct = [samp_to_dist(x, delay=delay) for x in samp_direct]
    echoes = [[samp_to_dist(x, delay=delay) for x in y] for y in samp_echoes]

    combinations = window_echoes(D, echoes)
    top_scoring_combos = []
    for i in range(len(combinations)):
        t1 = time()
        top_scoring_combos.append(score_echoes(D, combinations[i]))
        t2 = time()
        print("PROGESS: " + str(int((i+1)/len(combinations)*100)) + "%, last optimiztaion took " + str(int(t2-t1)) + "s")
    top_scoring_combos.sort()
    # for case when nPoints isn't reached, but it could be
    # why is this being done???
    if len(top_scoring_combos) < 201:
        top_scoring_combos.append((1e9, np.matrix([1e9]*D.shape[1])))
    else:
        top_scoring_combos[201] = (1e9, np.matrix([1e9]*D.shape[1]))
    # remove duplicates
    non_repeated_combos = [top_scoring_combos[0][1]]
    for i in range(1, len(top_scoring_combos)):
        if len(non_repeated_combos) >= 22: # nPoints
            return non_repeated_combos
        non_repeated_combos.append(top_scoring_combos[i][1])
        for j in range(0, i):
            if np.count_nonzero(top_scoring_combos[i][1] == top_scoring_combos[j][1]):
                non_repeated_combos.pop()
                break

    non_repeated_combos = non_repeated_combos[:-1]
    #pk.dump(non_repeated_combos, open('n_r_combos.txt', 'wb'))
    #non_repeated_combos = pk.load(open('n_r_combos.txt', 'rb'))

    dist, microphones, d_estim = optimization(np.square(D))

    nPoints = np.minimum(len(non_repeated_combos), 22)

    loudspeaker_loc, loudspeaker_error = trilaterate_back(microphones, np.reshape(np.array(direct), (1, 5)))
    estimated_images = np.zeros((3, nPoints))
    estimated_images_errors = np.zeros(nPoints)

    print('/--------------------' + "\\")
    print('|#src| dist/2 | error|')
    print('|--------------------|')
    for i in np.arange(nPoints):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimated_images[:,i], estimated_images_errors[i] = trilaterate_back(microphones, np.sqrt(np.array(non_repeated_combos[i])))
        print('|', i+1, '|', np.linalg.norm(estimated_images[:,i] - loudspeaker_loc)/2, '|', estimated_images_errors[i], '|')
    print('\\--------------------/')
    print('Error Threshold is set at:', 0.5)

    estimated_images_final = []
    estimated_images_errors_final = []
    for i in np.arange(nPoints):
        if(estimated_images_errors[i] <= 0.5):
            estimated_images_final.append(estimated_images[:,i].T.tolist())
            estimated_images_errors_final.append(estimated_images_errors[i])

    loudspeaker_loc = np.real(loudspeaker_loc)
    A, b, V = prune_echoes(np.array(estimated_images_final).T, loudspeaker_loc, 2)
    V = np.array([x[1:] for x in V])
    return V, loudspeaker_loc, microphones

# samp_to_dist
# input  - int: sample to convert to distance
#          int (optional) Fs: sampling rate (default 96khz)
#          int (optional) delay: delay for speaker (ask Dokmanic)
# output - float: distance to microphone
def samp_to_dist(sample, Fs=96000, delay=0):
    c = 343  # meters per second
    return (sample - delay) * c / Fs

# window_echoes
# input  - matrix: N x N matrix of microphone distances (unsquared)
#          list(list(distance)): size N list, each element is a list of echo distanes per mic
# output - list(list(distance)): size N list, each element is a list of echoes that could be related and should be scored
def window_echoes(D, echoes):
    windowSizeHalf = np.amax(D) * 1.3 # little bit extra for safety
    echoCombinations = []

    for mic1echo in echoes[0]:
        echoesLocal = [[mic1echo]]
        noGoodCombinationFlag = False
        for micNechoes in echoes[1:]:
            candidates = []
            for micNecho in micNechoes:
                if abs(micNecho - mic1echo) <= windowSizeHalf:
                    candidates.append(micNecho)
            if len(candidates) == 0:
                noGoodCombinationFlag = True
                break
            echoesLocal.append(candidates)
        if noGoodCombinationFlag:
            continue
        echoCombinations.append(echoesLocal)
    
    return echoCombinations

# score_echoes
# input  - matrix: N x N matrix of microphone distanecs (unsquared)
#          list(list(distance)): size N list, each element is a list of echoes that could be related and should be scored
# output - (int, matrix): tuple of (score, matrix) where matrix is anN x 1 matrix of associated echoes, one per mic
def score_echoes(D, echoTimes):
    
    combinationScores = []
    # construct all combinations of input echoes and square
    echoCombinations = combos_recursive(echoTimes, [])
    echoCombinations = [[y**2 for y in x] for x in echoCombinations]
    # we're using D.^2 for this, not just the distanes
    D = np.square(D)
    
    # score all combinations
    for echoCombo in echoCombinations:
        # create augmented matrix and score
        echoCombo = np.matrix(echoCombo)
        augmentedD = np.vstack([D, echoCombo])
        augmentedD = np.hstack([augmentedD, np.r_[echoCombo.transpose(),np.matrix([0])]])
        score,extra1,extra2 = optimization(augmentedD)
        combinationScores.append((score, echoCombo))
    combinationScores.sort()
    
    # Since this function will only be used with nPoints == 1, no need to filter out
    # repeated echoes
    return combinationScores[0]

# combos_recursive
# input  - list(list(distance)): list of possible echoes per mic
#          list(distance): current iterative list of echoes
# output - list(list(distance)): a list of all possible valid permutations
def combos_recursive(remainElems, permutation):
    if len(remainElems) == 0:
        return [permutation]
    solutions = []
    for elem in remainElems[0]:
        solutions += combos_recursive(remainElems[1:], permutation + [elem])
    return solutions

V, loudspeaker, microphones = roomshape(1)
fig = plt.figure()
fig.clf()
ax = Axes3D(fig)
ax.plot(V[:,0], V[:,1], V[:,2], 'o', c='red')
ax.plot(microphones[:,0], microphones[:,1], microphones[:,2], 'o', c='black')
ax.plot([loudspeaker[0]], [loudspeaker[1]], [loudspeaker[2]], 'o', c='blue')
plt.draw()
plt.show()
