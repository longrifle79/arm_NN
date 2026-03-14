import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
from helper import load_dataset


def initialize_zeros(dim):
    w,b = np.zeros((dim,1)),0
    return w,b


def sigmoid(x):
    y = 1.0/(1+np.exp(-x))
    return y


def propagation(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    
    cost  = -(1/m)*np.sum(Y*np.log(A)+ (1-Y)*np.log(1-A))
    
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    
    cost = np.squeeze(cost)
    grads ={ "dw" : dw,"db" : db}

    return grads,cost


def learn(w,b,X,Y,num_iterations,learning_rate):

    costs =[]

    for i in range(num_iterations):
        grads,cost = propagation(w,b,X,Y)
        costs.append(cost)

        w =  w - learning_rate *dw
        b =  b - learning_rate *db

        if i %50 ==0:
            print("Cost after iteration %i :  %f"%(i,cost))

        params ={"w" : w, "b":b}
        grads  ={"dw" : dw, "db" :db}

        return params,grads,costs
        
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

m_train  =  train_set_x_orig.shape[0]
m_test   =  test_set_x_orig.shape[0]
num_px   =  train_set_x_orig.shape[1]



train_set_x_flatten =  train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten    =  test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T













