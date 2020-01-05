#%% import module
import numpy as np
import time

#%% identity_function
def identity_function(x):
    
    return x

#%% step_function
def step_function(x):
    
    return np.array(x > 0, dtype = np.int)

#%% sigmoid
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

#%% sigmoid_grad
def sigmoid_grad(x):
    y = sigmoid(x)
    
    return y * (1 - y)

#%% relu
def relu(x):
    
    return np.maximum(0, x)

#%% relu_grad
def relu_grad(x):
    
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    
    return grad

#%% softmax
def softmax(X):
    if X.ndim == 1:
        M = np.max(X)
        X = X - M
        
        return np.exp(X) / (np.sum(np.exp(X)))
    elif X.ndim == 2:
        X = X.T
        M = np.max(X, axis = 0)
        X = X - M
        Y = np.exp(X) / np.sum(np.exp(X), axis = 0)
        Y = Y.T
        
        return Y
    else:
        print('3차원 이상 배열 연산 불가능!')
    
#%% mean_squared_error
def mean_squared_error(y, t):
    
    return 0.5 * np.sum((y - t) ** 2)

#%% cross_entropy_error
def cross_entropy_error(Y, T):
    if Y.ndim == 1:
        Y = Y.reshape(1, Y.size)
        T = T.reshape(1, T.size)
    
    if Y.shape == T.shape:
        T = np.argmax(T, axis = 1)
    
    batch_size = Y.shape[0]
    
    return -np.sum(np.log(Y[np.arange(batch_size), T] + 1e-7)) / batch_size

#%% softmax_loss
def softmax_loss(X, T):
    Y = softmax(X)
    l = cross_entropy_error(Y, T)
    
    return l

#%% numerical_gradient
def numerical_gradient(f, X):
    h = 1e-4
    grad = np.zeros_like(X)
    it = np.nditer(X, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = X[idx]
        
        X[idx] = float(tmp_val) + h
        fx_right = f(X)
        
        X[idx] = float(tmp_val) - h
        fx_left = f(X)
        
        grad[idx] = (fx_right - fx_left) / (2 * h)
        X[idx] = tmp_val
        it.iternext()
    
    return grad
