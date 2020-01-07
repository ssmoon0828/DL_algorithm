#%% import module
import numpy as np
from functions import *

#%% Relu
class Relu:
    
    def __init__(self):
        self.mask = None
        
    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dX = dout
        
        return dX
        
#%% Sigmoid
class Sigmoid:
    
    def __init__(self):
        self.out = None
        
    def forward(self, X):
        self.out = sigmoid(X)
        
        return self.out
    
    def backward(self, dout):
        dX = dout * self.out * (1.0 - self.out)
        
        return dX

#%% Affine
class Affine:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None
        
    def forward(self, X):
        self.X = X
        out = np.dot(X, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dX
    
#%% class SoftmaxWithLoss
class SoftmaxWithLoss:
    
    def __init__(self):
        self.T = None
        self.Y = None
        self.loss = None
        
    def forward(self, X, T):
        self.T = T
        self.Y = softmax(X)
        self.loss = cross_entropy_error(self.Y, self.T)
        
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.T.shape[0]
        dX = (self.Y - self.T) / batch_size
        
        return dX