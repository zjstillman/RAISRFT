import pickle
import argparse
from matplotlib import pyplot as plt
import numpy as np

filterpath = ''
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('filter', type = str, help = 'Which filter to display')
    args = parser.parse_args()
    filterpath = 'filters/' + args.filter

patchsize = 11
Qangle = 24
Qstrength = 3
Qcoherence = 3

with open(filterpath, "rb") as fp:
    h = pickle.load(fp)

f = plt.figure()

def wrap(arr):
	a = np.zeros((patchsize, patchsize))
		# , dtype = complex)
	for i in range(patchsize):
		for j in range(patchsize):
			a[i,j] = arr[i*patchsize + j]
	return a

for i in range(Qangle):
	for j in range(Qstrength):
		for k in range(Qcoherence):
			try:
				a = plt.subplot2grid((Qcoherence*Qstrength,Qangle), (j*Qcoherence + k,i))
				a.imshow(wrap(h[i, j, k, 0, 3]), cmap = 'gray')
				a.set_title(str(i) + ', ' + str(j) + ', ' + str(k))
				plt.axis('off')
			except IndexError:
				break

plt.suptitle('Angle, Strength, Coherence')

plt.show()