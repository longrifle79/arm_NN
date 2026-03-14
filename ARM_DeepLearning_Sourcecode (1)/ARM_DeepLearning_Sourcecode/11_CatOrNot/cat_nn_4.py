import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
from helper import load_dataset


def initialize_zeros(dim):
    w,b = np.zeros((dim,1)),0
    return w,b


weight,bias = initialize_zeros(3)

print("Weight = "+str(weight))
print("Bias = "+str(bias))

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

m_train  =  train_set_x_orig.shape[0]
m_test   =  test_set_x_orig.shape[0]
num_px   =  train_set_x_orig.shape[1]



train_set_x_flatten =  train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten    =  test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print("Train set X shape is : "+str(train_set_x_flatten.shape))
print("Test set X shape is : "+str(test_set_x_flatten.shape))
print("Train set y shape : " + str(train_set_y.shape))
print("Test set y shape : " + str(test_set_y.shape))











