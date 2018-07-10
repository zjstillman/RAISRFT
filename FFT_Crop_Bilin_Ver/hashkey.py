import numpy as np
from math import atan2, floor, pi
# from numba import jit

# @jit
def hashkey(block, Qangle, W, dubC = False, dubS = False):
    # Calculate gradient
    gy, gx = np.gradient(block)

    # Transform 2D matrix into 1D array
    gx = gx.ravel()
    gy = gy.ravel()

    # SVD calculation
    G = np.vstack((gx,gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG);

    # Make sure V and D contain only real numbers
    nonzerow = np.count_nonzero(np.isreal(w))
    nonzerov = np.count_nonzero(np.isreal(v))
    if nonzerow != 0:
        w = np.real(w)
    if nonzerov != 0:
        v = np.real(v)

    # Sort w and v according to the descending order of w
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    # Calculate theta
    theta = atan2(v[1,0], v[0,0])
    if theta < 0:
        theta = theta + pi
    # if theta > pi/2:
        # print(theta)
    # Calculate lamda
    lamda = w[0]

    # Calculate u
    sqrtlamda1 = np.sqrt(w[0])
    sqrtlamda2 = np.sqrt(w[1])
    if sqrtlamda1 + sqrtlamda2 == 0:
        u = 0
    else:
        u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)

    # Quantize
    angle = floor(theta/pi*Qangle)
    if dubS:
        if lamda < 0.00001:
            strength = 0
        elif lamda > 0.01:
            strength = 5
        elif lamda < .0001:
            strength = 1
        elif lamda > 0.001:
            strength = 4
        elif lamda < 0.0004:
            strength = 2
        else:
            strength = 3
    else:
        if lamda < 0.0001:
            strength = 0
        elif lamda > 0.001:
            strength = 2
        else:
            strength = 1
    # strength = 0
    if dubC:
        if u < 0.1:
            coherence = 0
        elif u < 0.25:
            coherence = 1
        elif u < 0.4:
            coherence = 2
        elif u < 0.55:
            coherence = 3
        elif u < 0.7:
            coherence = 4
        else:
            coherence = 5
    else:
        if u < 0.25:
            coherence = 0
        elif u > 0.5:
            coherence = 2
        else:
            coherence = 1

    # Bound the output to the desired ranges
    if angle > Qangle:
        angle = Qangle
    elif angle < 0:
        angle = 0

    return angle, strength, coherence
