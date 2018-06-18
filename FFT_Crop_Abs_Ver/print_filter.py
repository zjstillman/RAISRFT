import pickle
from matplotlib import pyplot as plt
import numpy as np
basictrain = 'filter'
full2train = 'filterfull2'
onetrain = 'onetrain'

patchsize = 11
Qangle = 24
Qstrength = 3
Qcoherence = 3

with open(full2train, "rb") as fp:
    h = pickle.load(fp)

f = plt.figure()

def wrap(arr):
	a = np.zeros((patchsize, patchsize))
	for i in range(patchsize):
		for j in range(patchsize):
			a[i,j] = arr[i*patchsize + j]
	return a

for i in range(Qangle//2):
	for j in range(Qstrength):
		for k in range(Qcoherence):
			try:
				a = plt.subplot2grid((Qcoherence*Qstrength,Qangle//2), (j*Qcoherence + k,i))
				a.imshow(wrap(h[i, j, k, 0, 0]), cmap = 'gray')
				a.set_title(str(i) + ', ' + str(j) + ', ' + str(k))
				plt.axis('off')
			except IndexError:
				break

plt.suptitle('Angle, Strength, Coherence')

plt.show()