import numpy as np
import os
import pickle
import ra
import fft
import argparse

from gaussian2d import gaussian2d
from hashkey import hashkey
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
testpath = ''
filterpath = ''
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--show', action = 'store_true', help = 'Displays the original, the basic upscale, and the filtered upscale for each image in the testing set.')
    parser.add_argument('--total', action = 'store_true', help = 'Displays the total MSE sums')
    parser.add_argument('testing_set', type = str, help = 'The set to test on.')
    parser.add_argument('filter', type = str, help = 'Which filter to use')
    args = parser.parse_args()
    testpath = '../Testing_Sets/' + args.testing_set
    filterpath = 'filters/' + args.filter

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

anglec = np.zeros(Qangle)
coherencec = np.zeros(Qcoherence)
locationc = np.zeros(Qlocation*Qlocation)
strengthc = np.zeros(Qstrength)

# Read filter from file
with open(filterpath, "rb") as fp:
    h = pickle.load(fp)

# @jit
def zeropad(arr):
    n = np.zeros(arr.shape, dtype = 'complex')
    for i in range(arr.shape[0]//4, 3*arr.shape[0]//4):
        for j in range(arr.shape[1]//4, 3*arr.shape[1]//4):
            n[i][j] = arr[i][j]
    return n

# @jit
def apply_filter(arr):
    heightHR, widthHR = arr.shape
    predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin), dtype = complex)
    operationcount = 0
    totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
    for row in range(margin, heightHR-margin):
        for col in range(margin, widthHR-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            operationcount += 1
            # Get patch
            patch = arr[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1].copy()
            patch = patch.ravel()
            # Get gradient block
            gradientblock = arr[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1].copy()
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            location = row//(heightHR//Qlocation)*Qlocation + col//(widthHR//Qlocation)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)

            location = 0
            # angle = 0
            # strength = 0
            # coherence = 0
            anglec[angle] += 1
            coherencec[coherence] += 1
            locationc[location] += 1
            strengthc[strength] += 1
            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,location,pixeltype])
    return predictHR



# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(testpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.ra')):
            imagelist.append(os.path.join(parent, filename))

TotalMSEsimp = 0
TotalMSEfilt = 0
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin_im = ra.read_ra(image)

    origin_fft = fft.fftc(origin_im)
    origin_fft_zero = zeropad(origin_fft)
    upscaledLR_im = fft.ifftc(origin_fft_zero)
    origin = abs(origin_im)
    upscaledLR = abs(upscaledLR_im)
    predictHR = apply_filter(upscaledLR)
    heightHR, widthHR = upscaledLR.shape
    print()
    print('anlge:')
    print(anglec)
    print('coherence:')
    print(coherencec)
    print('location:')
    print(locationc)
    print('strength:')
    print(strengthc)
    print()
    ############### COLORING STUFF ###############
    
    # Scale back to [0,255]
    # predictHR = cv2.normalize(predictHR.astype('float'), None, 0, 255, cv2.NORM_MINMAX)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 4, 1)
    # ax.imshow(grayorigin, cmap='gray', interpolation='none')
    # ax = fig.add_subplot(1, 4, 2)
    # ax.imshow(upscaledLR, cmap='gray', interpolation='none')
    # ax = fig.add_subplot(1, 4, 3)
    # ax.imshow(predictHR, cmap='gray', interpolation='none')
    # Bilinear interpolation on CbCr field
    result = np.zeros((heightHR, widthHR))
    # y = ycrcvorigin[:,:,0]
    # bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
    # result[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)
    # cr = ycrcvorigin[:,:,1]
    # bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    # result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)
    # cv = ycrcvorigin[:,:,2]
    # bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
    # result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)
    result[margin:heightHR-margin,margin:widthHR-margin] = predictHR
    # result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    # ax = fig.add_subplot(1, 4, 4)
    # ax.imshow(result, interpolation='none')
    # cv2.imwrite('results/' + os.path.splitext(os.path.basename(image))[0] + '_result.ra', result)
    
    ##############################################

    imagecount += 1
    o = origin
    s = upscaledLR
    r = result
    print()
    MSE = 0
    MSE2 = 0
    for a in range(heightHR-2*margin):
        a += margin
        for b in range(widthHR-2*margin):
            b += margin
            MSE += abs((o[a][b] - r[a][b])) ** 2
            MSE2 += abs((s[a][b] - o[a][b])) ** 2
    TotalMSEsimp += MSE2
    TotalMSEfilt += MSE
    print('Simple Upscale: ' + str(MSE2))
    print('Filter: ' + str(MSE))
    print('Percent of error in filter: ' + str(MSE/MSE2))
    if args.show:
        fig = plt.figure()
        a = fig.add_subplot(1,4,1)
        a.imshow(origin, cmap='gray')
        a = fig.add_subplot(1,4,2)
        a.imshow(upscaledLR, cmap='gray')
        a = fig.add_subplot(1,4,3)
        a.imshow(result, cmap='gray')
        # Uncomment the following line to visualize the process of RAISR image upscaling
        plt.show()

    if args.total:
        print('TOTAL Simple Upscale: ' + str(TotalMSEsimp))
        print('TOTAL Filter: ' + str(TotalMSEfilt))
        print('TOTAL Percent of error in filter: ' + str(TotalMSEfilt/TotalMSEsimp))

##### File Creation For Results #####

# f = open('results/FULL2_TRAIN_Error_FULL_1.txt','w')
# f.write('MSE: ' + str(MSE))
# print('MSE: ' + str(MSE))
# f.close()

#####################################

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
