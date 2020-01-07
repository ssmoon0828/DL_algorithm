#%% import module
import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import time

#%% SGD
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

#%% learning with SGD
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(784, 50, 10)
optimizer = SGD(lr = 0.1)

batch_size = 100
iter_num = int(x_train.shape[0] / batch_size)
lr = 0.1

start_time = time.time()

for j in range(10):
    for i in range(iter_num):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        
    
        optimizer.update(network.params, network.gradient(x_batch, t_batch))
        
        if (i % 100 == 0) or (i == 599):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(j, i, train_acc, test_acc)

end_time = time.time()

print(end_time - start_time)

#%% Momentum
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        # v에 가중치 변수 shape와 맞는 0 할당
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        # 모멘텀 수식 적용
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

#%% learning with Momentum
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(784, 50, 10)
optimizer = Momentum(lr = 0.1, momentum = 0.9)

batch_size = 100
iter_num = int(x_train.shape[0] / batch_size)
lr = 0.1

start_time = time.time()

for j in range(10):
    for i in range(iter_num):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        
    
        optimizer.update(network.params, network.gradient(x_batch, t_batch))
        
        if (i % 100 == 0) or (i == 599):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(j, i, train_acc, test_acc)

end_time = time.time()

print(end_time - start_time)

#%% AdaGrad
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
        
#%% learning with AdaGrad
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(784, 50, 10)
optimizer = AdaGrad(lr = 0.1)

batch_size = 100
iter_num = int(x_train.shape[0] / batch_size)
lr = 0.1

start_time = time.time()

for j in range(10):
    for i in range(iter_num):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        
    
        optimizer.update(network.params, network.gradient(x_batch, t_batch))
        
        if (i % 100 == 0) or (i == 599):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(j, i, train_acc, test_acc)

end_time = time.time()

print(end_time - start_time)

#%% Adam
class Adam:
    
    def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.iter = 0
    
    def update(self, params, grads):
        if self.m == None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        
        for key in params.keys():
            self.m[key] += (1.0 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1.0 - self.beta1) * (grads[key] **2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

#%% learning with Adam
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(784, 50, 10)
optimizer = Adam(lr = 0.01)

batch_size = 100
iter_num = int(x_train.shape[0] / batch_size)

start_time = time.time()

for j in range(10):
    for i in range(iter_num):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        
    
        optimizer.update(network.params, network.gradient(x_batch, t_batch))
        
        if (i % 100 == 0) or (i == 599):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(j, i, train_acc, test_acc)

end_time = time.time()

print(end_time - start_time)
