#%% 모듈장착
import numpy as np
import matplotlib.pyplot as plt

#%% 평균제곱오차
def mse(y, t):
    
    return 0.5 * np.dot(y - t, y - t)

# case1 : 정답을 2로 간주한 경우
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
mse(y, t) # 0.09750000000000003

# case2 : 정답을 7로 간주한 경우
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
mse(y, t) # 0.5975

#%% 교차 엔트로피 오차
def cee(y, t):
    delta = 1e-7
    
    return -np.sum(t * np.log(y + delta))

# case1 : 정답을 2로 간주한 경우
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
cee(y, t) # 0.510825457099338

# case2 : 정답을 7로 간주한 경우
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
cee(y, t) # 2.302584092994546

#%% 미니배치 학습
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = False)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

#%% (배치용) 교차 엔트로피 오차 구현
def cee(y, t): # one-hot encoding 일 경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cee(y, t): # one-hot encoding 이 아닐경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#%% 수치 미분
def numerical_diff(f, x):
    h = 1e-4
    
    return (f(x + h) - f(x - h)) / (2 * h)

def func(x):
    
    return x ** 2

def function_1(x):
    
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0, 20, 0.01)
y = function_1(x)
plt.plot(x, y)
plt.grid()
plt.show()

numerical_diff(function_1, 5)
numerical_diff(function_1, 10)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        tmp = x[i]
        
        # 1) x + h
        x[i] = tmp + h
        fx_right = f(x)
        
        # 2) x - h
        x[i] = tmp - h
        fx_left = f(x)
        
        grad[i] = (fx_right - fx_left) / (2 * h)
        x[i] = tmp
        
    return grad

def function_2(x):
    
    return x[0] ** 2 + x[1] ** 2

x = np.array([3.0, 4.0])
function_2(x)
numerical_gradient(function_2, x)

#%% 경사 하강법
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

x = np.array([-3.0, 4.0])
gradient_descent(function_2, x, lr = 0.1)

#%% 신경망에서의 기울기
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simple_net:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
#%%
net = simple_net()
net.W

x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)
t = np.array([0, 0, 1], dtype = np.float64)
net.loss(x, t)

def f(a):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)

f(3)
f(2)
f(net.W)

#%% 2층 신경망 클래스 구현 (1)
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class two_layer_net:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

#%% 2층 신경망 클래스 구현 (2)
net = two_layer_net(784, 100, 10)
net.params['W1'].shape
net.params['b1'].shape
net.params['W2'].shape
net.params['b2'].shape

x = np.random.rand(100, 784)
y = net.predict(x)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)
grads['W1'].shape
grads['b1'].shape
grads['W2'].shape
grads['b2'].shape

#%% 미니배치 학습 구현하기
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(784, 50, 10)

for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)