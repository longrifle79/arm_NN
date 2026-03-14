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

        dw = grads["dw"]
        db = grads["db"]
        
        w =  w - learning_rate *dw
        b =  b - learning_rate *db

        if i %50 == 0:
            costs.append(cost)
            
        if i %50 ==0:
            print("Cost after iteration %i :  %f"%(i,cost))

        params ={"w" : w, "b":b}
        grads  ={"dw" : dw, "db" :db}

    return params,grads,costs



def predict(w,b,X):
    m = X.shape[1]
    Y_prediction  =  np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction


def log_reg_model(X_train,Y_train,X_test,Y_test,num_iterations = 1000,learning_rate =0.4):

    w,b = initialize_zeros(X_train.shape[0])
    parameters,grads,costs = learn(w,b,X_train,Y_train,num_iterations,learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test  = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    np.mean((Y_prediction_test - Y_test))
    
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100))
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train))*100))

    d = { "costs" :costs, "Y_prediction_test" :Y_prediction_test,"Y_prediction_train":Y_prediction_train,"w":w,"b":b,"learning_rate":learning_rate,"num_iterations":num_iterations}

    return d


#------Preprocessing----------------#    
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

m_train  =  train_set_x_orig.shape[0]
m_test   =  test_set_x_orig.shape[0]
num_px   =  train_set_x_orig.shape[1]



train_set_x_flatten =  train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten    =  test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#---Normalize data-----#
train_set_x  =  train_set_x_flatten/255
test_set_x =  test_set_x_flatten/255

#--Training---#

d  = log_reg_model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations =5000,learning_rate = 0.005)

costs  = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per fifty)')
plt.title("Learning Rate "+str(d["learning_rate"]))
plt.show()








