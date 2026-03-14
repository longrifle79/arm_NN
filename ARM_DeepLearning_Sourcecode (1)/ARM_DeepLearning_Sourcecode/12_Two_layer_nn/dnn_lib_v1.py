import numpy as np


def initialize_parameters(n_x,n_h,n_y):

    np.random.seed(1)

    W1 = np.random.randn(n_h,n_x)*0.01
    W2 = np.random.randn(n_y,n_h)*0.01

    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))


    parameters = {"W1" :W1,"W2":W2,"b1":b1,"b2":b2}

    return parameters

def linear_forward(A,W,b):
    Z = W.dot(A)+b
    cache =(A,W,b)
    return Z,cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A,cache  

def sigmoid(Z):
    A  = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def linear_activation_forward(A_prev,W,b,activation):
    
    if activation == "sigmoid":
        
        Z,linear_cache =linear_forward(A_prev,W,b) 
        A,activation_cache  = sigmoid(Z)

    elif activation == "relu":
        Z,linear_cache =linear_forward(A_prev,W,b) 
        A,activation_cache = relu(Z)

    cache  = (linear_cache,activation_cache)
    return A,cache

def compute_cost(yHat,Y):
    m = Y.shape[1]
    cost  = (1./m)*(-np.dot(Y,np.log(yHat).T) -  np.dot(1-Y,np.log(1-yHat).T))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b =cache
    m = A_prev.shape[1]

    dW  = 1./m*np.dot(dZ,A_prev.T)
    db  = 1./m*np.sum(dZ,axis=1,keepdims = True)
    dA_prev =  np.dot(W.T,dZ)

    return dA_prev,dW,db

def relu_backward(dA,cache):
    Z =cache
    dZ =  np.array(dA,copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA,cache):
    Z =  cache
    s = 1/(1+np.exp(-Z))
    dZ =  dA*s*(1-s)
    return dZ
    
    
def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache =cache

    if activation  == "relu":
        
        dZ =  relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    elif activation  == "sigmoid":
        
        dZ =  sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)] =  parameters['W'+str(l+1)]- learning_rate*grads['dW' +str(l+1)]
        parameters['b'+str(l+1)] =  parameters['b'+str(l+1)]- learning_rate*grads['db' +str(l+1)]
    return parameters



def model_forward_prop(X,parameters):
    caches =[]
    A =X
    L = len(parameters)//2

    for l in range(1,L):
        A_prev =A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation ="relu")
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation ="sigmoid")
    caches.append(cache)

    return AL,caches
    
def predict(X,y,parameters):

    m = X.shape[1]
    n = len(parameters)//2
    p =  np.zeros((1,m))

    probabilities,caches  = model_forward_prop(X,parameters);

    for i in range(0,probabilities.shape[1]):
        if probabilities[0,i] > 0.5:
            p[0,i] =1
        else:
            p[0,i] =0
    print("Accuracy: "+str(np.sum((p==y)/m)))

    return p

    

    
    
    
    

















