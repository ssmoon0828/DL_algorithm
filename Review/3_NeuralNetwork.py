#%% 계단함수
def step_function(x):
    if x <= 0:
        return 0
    else:
        return 1

step_function(5)
step_function(-4)

#%% 계단함수(numpy 배열로 입력 가능)
import numpy as np

def step_function(x):
    tmp = x > 0
    
    return tmp.astype(np.int)

step_function(np.array([-4, 5]))

#%% 계단함수의 그래프
import matplotlib.pyplot as plt

def step_function(x):
    
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5, 5, 0.01)
y = step_function(x)

plt.plot(x, y)
plt.grid()

#%% 시그모이드 함수
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()

#%% 계단함수와 시그모이드 함수 비교
plt.plot(x, step_function(x), ls = '--', label = 'step_function')
plt.plot(x, sigmoid(x), label = 'sigmoid')
plt.legend()
plt.grid()

#%% 렐루(ReLU) 함수
def relu(x):
    return np.maximum(0, x)

plt.plot(x, relu(x))
plt.grid()

#%% 다차원 배열의 계산
A = np.array([1, 2, 3, 4])
print(A)
A.ndim
A.shape

B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
B.ndim
B.shape

#%% 행렬의 곱
A = np.array([[1, 2],
              [3, 4]])
A.shape

B = np.array([[5, 6],
              [7, 8]])
B.shape

np.dot(A, B)

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A.shape

B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
B.shape

np.dot(A, B)

#%% 신경망에서 행렬곱
x = np.array([1, 2])
x.shape

W = np.array([[1, 3, 5],
              [2, 4, 6]])
W.shape

y = np.dot(x, W)
y.shape

#%% 3층 신경망 구현하기
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],
                             [0.2, 0.4, 0.6]])
    network['b1'] = np.array([[0.1, 0.2, 0.3]])
    network['W2'] = np.array([[0.1, 0.4],
                             [0.2, 0.5],
                             [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3],
                             [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward_network(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    y = np.dot(z2, W3) + b3
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward_network(network, x)

#%% 소프트맥스 함수
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y)
np.sum(y)

#%% 소프트맥스 함수 구현
def softmax(a):
    c = np.mean(a)
    
    return np.exp(a - c) / np.sum(np.exp(a - c))

a = np.array([1010, 1000, 990])
softmax(a)

#%% 손글자 숫자 인식(mnist data)

#%% 데이터 불러오기
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

#%% 이미지 확인
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
img_show(img)
plt.imshow(img, cmap = 'gray')

#%% 신경망의 추론 처리
def get_data():
    (x_train, t_train) , (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    
    return x_test, t_test

import pickle

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()
accuracy = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    
    if p == t[i]:
        accuracy += 1

print('Accracy : {}'.format(accuracy / len(x)))

#%% 배치 처리
x.shape
print(x[0].shape)
print(network['W1'].shape)
print(network['W2'].shape)
print(network['W3'].shape)

# 배치 : 하나로 묶은 입력 데이터
# 이미지 1장(1, 784), 1장(1, 784), 1장(1, 784), 예측하면 속도가 너무 오래걸린다.
# 연산속도를 빠르게 하기위해 여러장을 하나로 묶어(가령 100장) 입력데이터(100, 784)를 만든다.

x, t = get_data()
network = init_network()
batch_size = 100
accuracy = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy += np.sum(p == t[i : i + batch_size])

print('Accuracy : {}'.format(accuracy / len(x)))
