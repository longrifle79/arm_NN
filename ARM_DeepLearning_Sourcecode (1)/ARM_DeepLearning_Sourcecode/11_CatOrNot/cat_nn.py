import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
from helper import load_dataset


train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

i = 1
print("y value is " + str(train_set_y[:,i]))
plt.imshow(train_set_x_orig[i])
plt.show()

