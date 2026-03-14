import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from helper import load_dataset
from dnn_lib_v1 import*
from scipy import ndimage

np.random.seed(1)

train_x_orig,train_y,test_x_orig,test_y,classes = load_dataset()

m_train  =  train_x_orig.shape[0]
num_px   =  train_x_orig.shape[1]
m_test   =  test_x_orig.shape[0]

##print("Training examples : " +str(m_train))
##print("Test examples : "+str(m_test))
##print("Image size : "+str(num_px)+","+str(num_px)+",3")
##print("Train_x_orig_shape " +str(train_x_orig.shape))
##print("Train_y_shape " +str(train_y.shape))
##print("Test_x_orig_shape " +str(test_x_orig.shape))
##print("Test_y_shape " +str(test_y.shape))

#Reshape images
train_x_flatten  = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten   = test_x_orig.reshape(test_x_orig.shape[0],-1).T

#Normalize images
train_x  = train_x_flatten/255
test_x    = test_x_flatten/255

n_x  = num_px * num_px * 3
n_h  = 6
n_y  = 1

layers_dims = (n_x,n_h,n_y)



def two_layer_nn_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations =3500):

    grads ={}
    costs =[]
    m = X.shape[1]
    (n_x,n_h,n_y) = layers_dims
    
    parameters = initialize_parameters(n_x,n_h,n_y)

    W1  = parameters["W1"]
    b1  = parameters["b1"]
    W2  = parameters["W2"]
    b2  = parameters["b2"]

    for i in range(0,num_iterations):
        A1,chache1 = linear_activation_forward(X,W1,b1,'relu')
        A2,chache2 = linear_activation_forward(A1,W2,b2,'sigmoid')
        
        cost = compute_cost(A2,Y)

        dA2  = -(np.divide(Y,A2) -  np.divide(1-Y,1-A2))
        dA1,dW2,db2 = linear_activation_backward(dA2,chache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,chache1,"relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        
        parameters = update_parameters(parameters,grads,learning_rate)

        W1 =  parameters["W1"]
        b1 =  parameters["b1"]
        W2 =  parameters["W2"]
        b2 =  parameters["b2"]

        if i%100 ==0:
            print("Cost after iteration {}: {}".format(i,np.squeeze(cost)))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" +str(learning_rate))
    plt.show()

    return parameters
 
        
    
parameters = two_layer_nn_model(train_x,train_y,layers_dims = (n_x,n_h,n_y),num_iterations = 2500)
    
train_predciton  = predict(train_x,train_y,parameters)
test_prediction  = predict(test_x,test_y,parameters)



def print_wrong_predictions(classes,X,y,p):

    a  = p +y

    mislabeled_indices =  np.asarray(np.where(a==1))
    plt.rcParams['figure.figsize'] = (50.0,50.0)
    num_images =  len(mislabeled_indices[0])

    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(2,num_images,i+1)
        plt.imshow(X[:,index].reshape(64,64,3),interpolation ='nearest')
        plt.axis('off')
        plt.title("Prediction: "+classes[int(p[0,index])].decode("utf-8")+"\n Class :"+classes[y[0,index]].decode("utf-8"))


 #Uncomment to show wrong predictions
        
#print_wrong_predictions(classes,test_x,test_y,test_prediction)
#plt.show()

my_label_y = [0]
my_image = "img6.jpg"
fname  = "images/"+my_image
image = np.array(ndimage.imread(fname,flatten=False))
my_img = scipy.misc.imresize(image,size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
img_prediction = predict(my_img,my_label_y,parameters)

plt.imshow(image)
print("The prediction is "+str(np.squeeze(img_prediction)))
plt.show()









    
    










