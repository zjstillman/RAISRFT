import os
import ra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

folder = 'train_img_slices'
filename = '1_0.ra'

img = ra.read_ra(os.path.join(folder, filename))
mag = abs(img)

plt.figure()
plt.imshow(mag, cmap='gray')
plt.show()
