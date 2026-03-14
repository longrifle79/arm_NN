import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
from helper import load_dataset


train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

##---Test Point 1-----###
##i = 1
##print("y value is " + str(train_set_y[:,i]))
##plt.imshow(train_set_x_orig[i])
##plt.show()

# train_set_x_orig = ( number_of_images ,height,width,3)
#train_set_x_orig  =  (m_train,num_px,num_px,3)

m_train  =  train_set_x_orig.shape[0]
m_test   =  test_set_x_orig.shape[0]
num_px   =  train_set_x_orig.shape[1]

print(" No of train egs : " + str(m_train))
print(" No of test egs : " + str(m_test))
print(" Image height and width  : " + str(num_px))

print("Train set x shape : " + str(train_set_x_orig.shape))
print("Train set y shape : " + str(train_set_y.shape))

print("Test set x shape : " + str(test_set_x_orig.shape))
print("Test set y shape : " + str(test_set_y.shape))




