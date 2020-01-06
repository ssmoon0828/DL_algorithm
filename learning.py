#%% import module
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from time import time

#%% load data, network
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True,
                                                  one_hot_label = True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

#%% verify gradient with numerical gradient
x_batch = x_train[:3]
t_batch = t_train[:3]

start_time = time()
grad_numerical = network.numerical_gradient(x_batch, t_batch)
end_time = time()
time_numerical = end_time - start_time
print(time_numerical)

start_time = time()
grad_backprop = network.gradient(x_batch, t_batch)
end_time = time()
time_backprop = end_time - start_time
print(time_backprop)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))


#%% learning
batch_size = 100
iter_num = int(x_train.shape[0] / batch_size)
learning_rate = 0.1

start_time = time()
for j in range(17):
    for i in range(iter_num):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    print(j, train_acc, test_acc)

end_time = time()
print(end_time - start_time)
