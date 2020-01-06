#%% import module
import numpy as np
from functions import *
from layers import *
from collections import OrderedDict

#%% two_layer_net
class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, init_weight_std = 0.01):
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = init_weight_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = init_weight_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, X):
        
        for layer in self.layers.values():
            X = layer.forward(X)
        
        return X
    
    def loss(self, X, T):
        Y = self.predict(X)
        
        return self.last_layer.forward(Y, T)
    
    def accuracy(self, X, T):
        Y = self.predict(X)
        Y = np.argmax(Y, axis = 1)
        
        if T.ndim == 2:
            T = np.argmax(T, axis = 1)
        
        return np.sum(Y == T) / float(X.shape[0])
    
    def numerical_gradient(self, X, T):
        loss_W = lambda W : self.loss(X, T)
        
        grad = {}
        grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grad
    
    def gradient(self, X, T):
        self.loss(X, T) # 순전파를 한번 걸어주어야한다
        
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        grad = {}
        grad['W1'] = self.layers['Affine1'].dW
        grad['b1'] = self.layers['Affine1'].db
        grad['W2'] = self.layers['Affine2'].dW
        grad['b2'] = self.layers['Affine2'].db
        
        return grad