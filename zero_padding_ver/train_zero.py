import cv2
import numpy as np
import os
import pickle
import ra
import fft
# from numba import jit
from scipy.misc import imresize
from cgls_im import cgls
from filterplot_zero import filterplot
from gaussian2d_zero import gaussian2d
from hashkey_zero import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate

# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
Qlocation = 3
trainpath = 'train'
trainlarge = 'train_large'
trainfull2path = 'train_full_2'
basictrain = 'filter'
full2train = 'filterfull2'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

Q = np.zeros((Qangle, Qstrength, Qcoherence, Qlocation*Qlocation, R*R, patchsize*patchsize, patchsize*patchsize), dtype = complex)
V = np.zeros((Qangle, Qstrength, Qcoherence, Qlocation*Qlocation, R*R, patchsize*patchsize), dtype = complex)
h = np.zeros((Qangle, Qstrength, Qcoherence, Qlocation*Qlocation, R*R, patchsize*patchsize), dtype = complex)
mark = np.zeros((Qstrength, Qcoherence, Qangle, Qlocation*Qlocation, R*R))
anglec = np.zeros(Qangle)
coherencec = np.zeros(Qcoherence)
locationc = np.zeros(Qlocation*Qlocation)
strengthc = np.zeros(Qstrength)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# @jit
def zeropad(arr):
    n = np.zeros(arr.shape, dtype = complex)
    for i in range(arr.shape[0]//6, 5*arr.shape[0]//6):
        for j in range(arr.shape[1]//6, 5*arr.shape[1]//6):
            n[i][j] = arr[i][j]
    return n

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainfull2path):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.ra')):
            imagelist.append(os.path.join(parent, filename))

# Compute Q and V
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rProcessing image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin = ra.read_ra(image)
    
    origin_fft = fft.fftc(origin)
    origin_fft_zero = zeropad(origin_fft)
    upscaledLR = fft.ifftc(origin_fft_zero)

    # Calculate A'A, A'b and push them into Q, V
    height, width = upscaledLR.shape
    operationcount = 0
    totaloperations = (height-2*margin) * (width-2*margin)
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            # print(patch)
            patch = np.matrix(patch.ravel())
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            location = row//(height//Qlocation)*Qlocation + col//(width//Qlocation)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)

            location = 0
            # angle = 0
            # strength = 0
            # coherence = 0

            # Get corresponding HR pixel
            pixelHR = origin[row,col]
            # Compute A'A and A'b
            ATA = np.dot(patch.T.conjugate(), patch)
            # print(ATA)
            ATb = np.dot(patch.T.conjugate(), pixelHR)
            ATb = np.array(ATb).ravel()
            # Compute Q and V
            Q[angle,strength,coherence,location,pixeltype] += ATA
            V[angle,strength,coherence,location,pixeltype] += ATb
            mark[coherence, strength, angle, location, pixeltype] += 1
            anglec[angle] += 1
            coherencec[coherence] += 1
            locationc[location] += 1
            strengthc[strength] += 1
    imagecount += 1
print()
# print (mark)
print('anlge:')
print(anglec)
print('coherence:')
print(coherencec)
print('location:')
print(locationc)
print('strength:')
print(strengthc)
print()
# Preprocessing permutation matrices P for nearly-free 8x more learning examples
# print('\r', end='')
# print(' ' * 60, end='')
# print('\rPreprocessing permutation matrices P for nearly-free 8x more learning examples ...')
# P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7), dtype = complex)
# rotate = np.zeros((patchsize*patchsize, patchsize*patchsize), dtype = complex)
# flip = np.zeros((patchsize*patchsize, patchsize*patchsize), dtype = complex)
# for i in range(0, patchsize*patchsize):
#     i1 = i % patchsize
#     i2 = floor(i / patchsize)
#     j = patchsize * patchsize - patchsize + i2 - patchsize * i1
#     rotate[j,i] = 1+0j
#     k = patchsize * (i2 + 1) - i1 - 1
#     flip[k,i] = 1+0j
# for i in range(1, 8):
#     i1 = i % 4
#     i2 = floor(i / 4)
#     P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
# Qextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize), dtype = complex)
# Vextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize), dtype = complex)
# for pixeltype in range(0, R*R):
#     for angle in range(0, Qangle):
#         for strength in range(0, Qstrength):
#             for coherence in range(0, Qcoherence):
#                 for m in range(1, 8):
#                     m1 = m % 4
#                     m2 = floor(m / 4)
#                     newangleslot = angle
#                     if m2 == 1:
#                         newangleslot = Qangle-angle-1
#                     newangleslot = int(newangleslot-Qangle/2*m1)
#                     while newangleslot < 0:
#                         newangleslot += Qangle
#                     newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
#                     newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
#                     Qextended[newangleslot,strength,coherence,pixeltype] += newQ
#                     Vextended[newangleslot,strength,coherence,pixeltype] += newV
# Q += Qextended
# V += Vextended

# Compute filter h
# @jit
def compute_filter_pixel(anlge, strength, coherence, location, pixeltype, Q, V):
    return np.linalg.lstsq(Q[angle,strength,coherence,location,pixeltype], V[angle,strength,coherence,location,pixeltype], rcond = 1e-3)[0]

print('Computing h ...')
operationcount = 0
totaloperations = R * R * Qangle * Qstrength * Qcoherence * Qlocation*Qlocation
print(totaloperations)
for pixeltype in range(0, R*R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                for location in range(0, Qlocation*Qlocation):
                    print('\r' + str(operationcount) + ' '*100, end= '')
                    if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                        print('\r|', end='')
                        print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                        print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                        print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                    operationcount += 1
                    h[angle,strength,coherence,location,pixeltype] = np.linalg.lstsq(Q[angle,strength,coherence,location,pixeltype], V[angle,strength,coherence,location,pixeltype], rcond = 1e-3)[0]

# Write filter to file
with open(full2train, "wb") as fp:
    pickle.dump(h, fp)

# Uncomment the following line to show the learned filters
# filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
