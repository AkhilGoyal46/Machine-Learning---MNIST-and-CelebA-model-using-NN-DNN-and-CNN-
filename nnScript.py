import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.utils import shuffle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return pow((1+ np.exp(-1*z)),-1)

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_data=np.array(mat['train0'])
    train_label=np.zeros([np.shape(mat['train0'])[0],1])
    
    train_data=np.vstack((train_data,mat['train1']))
    train_label=np.vstack((train_label,np.ones([np.shape(mat['train1'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train2']))
    train_label=np.vstack((train_label,2 * np.ones([np.shape(mat['train2'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train3']))
    train_label=np.vstack((train_label,3 * np.ones([np.shape(mat['train3'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train4']))
    train_label=np.vstack((train_label,4 * np.ones([np.shape(mat['train4'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train5']))
    train_label=np.vstack((train_label,5 * np.ones([np.shape(mat['train5'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train6']))
    train_label=np.vstack((train_label,6 * np.ones([np.shape(mat['train6'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train7']))
    train_label=np.vstack((train_label,7 * np.ones([np.shape(mat['train7'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train8']))
    train_label=np.vstack((train_label,8 * np.ones([np.shape(mat['train8'])[0],1])))
    
    train_data=np.vstack((train_data,mat['train9']))
    train_label=np.vstack((train_label,9 * np.ones([np.shape(mat['train9'])[0],1])))
    
    
    test_data=np.array(mat['test0'])
    test_label=np.zeros([np.shape(mat['test0'])[0],1])
    
    test_data=np.vstack((test_data,mat['test1']))
    test_label=np.vstack((test_label,np.ones([np.shape(mat['test1'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test2']))
    test_label=np.vstack((test_label,2 *np.ones([np.shape(mat['test2'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test3']))
    test_label=np.vstack((test_label,3 * np.ones([np.shape(mat['test3'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test4']))
    test_label=np.vstack((test_label,4 * np.ones([np.shape(mat['test4'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test5']))
    test_label=np.vstack((test_label,5 * np.ones([np.shape(mat['test5'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test6']))
    test_label=np.vstack((test_label,6 * np.ones([np.shape(mat['test6'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test7']))
    test_label=np.vstack((test_label,7 * np.ones([np.shape(mat['test7'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test8']))
    test_label=np.vstack((test_label,8 * np.ones([np.shape(mat['test8'])[0],1])))
    
    test_data=np.vstack((test_data,mat['test9']))
    test_label=np.vstack((test_label,9 * np.ones([np.shape(mat['test9'])[0],1])))
    
    data= np.vstack((train_data,test_data))
    
    
    train_data, train_label = shuffle(train_data, train_label, random_state=0)
    
    
    
    validation_data=train_data[0:10000,:]
    validation_label=train_label[0:10000,:]
    
    train_data=train_data[10000:,:]
    train_label=train_label[10000:,:]
    
    
    # Feature selection
    # Your code here.
    suma = data.sum(axis=0)
    idx = np.where(suma!=0)
    train_data=np.squeeze(train_data[:,idx],axis=1)
    test_data=np.squeeze(test_data[:,idx],axis=1)
    validation_data=np.squeeze(validation_data[:,idx],axis=1)
    
    print('preprocess done')

    return train_data/255, train_label, validation_data/255, validation_label, test_data/255, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    training_data=np.hstack((np.ones([np.shape(training_data)[0],1]),training_data))
    hidden_nodes = sigmoid(training_data@np.transpose(w1))
    hidden_nodes =np.hstack((np.ones([np.shape(hidden_nodes)[0],1]),hidden_nodes))
    output= sigmoid(hidden_nodes@np.transpose(w2))
    
    label = np.zeros( [len(training_label), n_class])
    for i in range(len(training_label)):
        label[ i, int(training_label[i]) ] = 1
    
    obj_val = lambdaval * (np.sum(pow(w1,2)) + np.sum(pow(w2,2)))/2
    
    obj_val += -1*((label.flatten().transpose()@np.log(output.flatten().transpose()))+(1-label.flatten().transpose())@np.log(1-output.flatten().transpose()))
    #for i in range(len(training_label)):
     #   for k in range(n_class):
     #      obj_val -= label[i,k]*np.log(output[i,k]) + (1 - label[i,k])*np.log(1 - output[i,k])
    
    obj_val = obj_val/len(training_label)
    grad_w1 = lambdaval * w1
    grad_w2 = lambdaval * w2
    
    
    
    b= np.transpose(np.multiply( np.multiply(1 - hidden_nodes,hidden_nodes), ((output-label)@w2)))
    
    grad_w1 += (b@training_data)[1:,:]
    
    #for i in range(len(training_label)):
    #   for j in range(n_input):
     #       for k in range(n_hidden):
     #           a=0
     #           for l in range(n_class):
      #              a += (output[i,l] - label[i,l])*w2[l,k]
      #          grad_w1[k,j] += (1 - hidden_nodes[i,k])*hidden_nodes[i,k]*a*training_data[i,j]
    
    
    
    grad_w2 += np.transpose(output - label) @ hidden_nodes
    
    grad_w1 = grad_w1/len(training_label)
    grad_w2 = grad_w2/len(training_label)
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    
    # Your code here
    data=np.hstack((np.ones([np.shape(data)[0],1]),data))
    hidden_nodes = sigmoid(data@np.transpose(w1))
    hidden_nodes =np.hstack((np.ones([np.shape(hidden_nodes)[0],1]),hidden_nodes))
    output= sigmoid(hidden_nodes@np.transpose(w2))
    labels=np.argmax(output,axis=1)
    labels = labels.reshape([np.shape(data)[0],1])
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50} 

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
