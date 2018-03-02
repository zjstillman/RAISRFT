import cv2
import numpy as np
import os
import pickle
import ra
import fft
from gaussian2d_im import gaussian2d
from hashkey_im import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
# Added:
from skimage.transform import resize
from scipy.misc import imresize

def downsample(arr):
    n = np.zeros((arr.shape[0]//2, arr.shape[1]//2), dtype = complex)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if(i%2 == 0 and j%2 == 0):
                n[i//2][j//2] = arr[i][j]
    return n

# Define parameters
R = 1
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
Qlocation = 3
trainpath = 'test'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

# Read filter from file
with open("filter", "rb") as fp:
    h = pickle.load(fp)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.ra')):
            imagelist.append(os.path.join(parent, filename))


imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin_nofft = ra.read_ra(image)
    ####### Normalizing, not sure if needed #######
    # origin_norm = origin_nofft / max(np.absolute(origin_nofft).ravel())
    origin_norm = origin_nofft
    ###############################################
    origin_fft = fft.fftc(origin_norm)
    ### Added code ###
    height, width= origin_norm.shape
    origin = downsample(origin_fft)
    ##################


    

    # Upscale (bilinear interpolation)
    or_re = np.real(origin)
    or_im = np.imag(origin)
    heightLR, widthLR = or_re.shape
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, or_re, kind='linear')
    heightgridHR = np.linspace(0,heightLR-1,heightLR*2-1)
    widthgridHR = np.linspace(0,widthLR-1,widthLR*2-1)
    upscaledLR_re = bilinearinterp(widthgridHR, heightgridHR)

    heightLR, widthLR = or_im.shape
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, or_im, kind='linear')
    heightgridHR = np.linspace(0,heightLR-1,heightLR*2-1)
    widthgridHR = np.linspace(0,widthLR-1,widthLR*2-1)
    upscaledLR_im = bilinearinterp(widthgridHR, heightgridHR)
    # Calculate predictHR pixels
    upscaledLR = np.add(upscaledLR_re, np.multiply(0+1j, upscaledLR_im, dtype = complex), dtype = complex)

    heightHR, widthHR = upscaledLR.shape
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
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = patch.ravel()
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            location = row//(heightHR//3)*Qlocation + col//(widthHR//3)

            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)

            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,location,pixeltype])
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
    result = np.zeros((heightHR, widthHR), dtype = complex)
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
    imagecount += 1

    ### Added code ###
    # cv2.imshow('Original FFT', np.absolute(origin_fft))
    # cv2.imshow('Original', np.absolute(origin_norm))
    # cv2.imshow('Original after reversed FFT', np.absolute(fft.ifftc(origin_fft)))
    # cv2.imshow('Original FFT scaled down', np.absolute(origin))
    # # cv2.imshow('Gray down', grayorigin)
    # cv2.imshow('Upscaled simple', np.absolute(upscaledLR))
    # cv2.imshow('Upscaled simple reversed', np.absolute(fft.ifftc(upscaledLR)))
    # cv2.imshow('Result', np.absolute(result))
    # cv2.imshow('Result image', np.absolute(fft.ifftc(result)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ##################

    # Uncomment the following line to visualize the process of RAISR image upscaling
    fig = plt.figure()
    a = fig.add_subplot(1,4,1)
    a.imshow(abs(fft.ifftc(origin_fft)), cmap='gray')
    a = fig.add_subplot(1,4,2)
    a.imshow(abs(fft.ifftc(upscaledLR)), cmap='gray')
    a = fig.add_subplot(1,4,3)
    a.imshow(abs(fft.ifftc(result)), cmap='gray')
    plt.show()
    # plt.show()


print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
