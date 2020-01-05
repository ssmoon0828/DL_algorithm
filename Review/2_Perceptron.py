#%% AND gate 구현
def and_gate(x1, x2):
    tmp = (2 * x1) + (2 * x2) - 3
    
    if tmp <= 0:
        return 0
    else:
        return 1

for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, and_gate(x1, x2)))
        
#%% NAND gate 구현
def nand_gate(x1, x2):
    tmp = (2 * x1) + (2 * x2) - 3
    
    if tmp <= 0:
        return 1
    else:
        return 0
    
for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, nand_gate(x1, x2)))
        
#%% OR gate 구현
def or_gate(x1, x2):
    tmp = (2 * x1) + (2 * x2) - 1
    
    if tmp <= 0:
        return 0
    else:
        return 1

for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, or_gate(x1, x2)))
        
#%% 가중치와 편향 도입
import numpy as np

x = np.array([0, 1])
w = np.array([2, 2])
b = -3

np.dot(w, x) + b

#%% AND gate 구현
def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([2, 2])
    b = -3
    
    tmp = np.dot(w, x) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1

for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, and_gate(x1, x2)))

#%% NAND gate 구현
def nand_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([2, 2])
    b = -3
    
    tmp = np.dot(w, x) + b
    
    if tmp <= 0:
        return 1
    else:
        return 0

for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, nand_gate(x1, x2)))

#%% OR gate 구현
def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([2, 2])
    b = -1
    
    tmp = np.dot(w, x) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, or_gate(x1, x2)))

#%% XOR gate 구현
def xor_gate(x1, x2):
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    y = and_gate(s1, s2)
    
    return y
    
for x2 in range(0, 2):
    for x1 in range(0, 2):
        print('x1 : {}, x2 : {}, y : {}'.format(x1, x2, xor_gate(x1, x2)))
