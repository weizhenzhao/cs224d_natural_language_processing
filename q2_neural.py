'''
Created on Sep 11, 2017

@author: p0079482
'''

"""写出只有一个隐藏层其激活函数为sigmoid的神经网络前向和后向传播代码"""
import numpy as np
from q1_sigmoid import sigmoid
from q1_softmax import softmax
from q2_sigmoid import sigmoid_graf
def forward_backward_prop(data, labels, params, verbose=False):
    """2 个隐层的神经网络的前向运算和反向传播"""
    
    if len(data.shape) >= 2:
        (N, _) = data.shape
    
    dimensions = [1] * 3
    dimensions[0]=N
    dimensions[1]=20
    dimensions[2]=labels.shape
    
    
    # ## 展开每一层神经网络的参数
    t = 0
    W1 = np.reshape(params[t:t + dimensions[0] * dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0] * dimensions[1]
    b1 = np.reshape(params[t:t + dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t + dimensions[1] * dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1] * dimensions[2]
    b2 = np.reshape(params[t:t + dimensions[2]], (1, dimensions[2]))
    
    # ##前向运算
    
    # 第一个隐层做内积
    a1 = sigmoid(data.dot(W1) + b1)
    # 第二个隐层做内积
    a2 = softmax(a1.dot(W2) + b2)
    
    cost = -np.sum(np.log(a2[labels == 1])) / N
    
    # ## 反向传播
    
    # Calculate analytic gradient for the cross entropy function
    grad_a2 = (a2 - labels) / N
    
    # Backpropagate through the second latent layer
    gradW2 = np.dot(a1.T, grad_a2)
    gradb2 = np.sum(grad_a2, axis=0, keepdims=True)
    
    # Backpropagate through the first latent layer
    grad_a1 = np.dot(grad_a2, W2.T) * sigmoid_graf(a1)
    
    gradW1 = np.dot(data.T, grad_a1)
    gradb1 = np.sum(grad_a1, axis=0, keepdims=True)
    
    if verbose:  # Verbose mode for logging information
        print("W1 shape:{}".format(str(W1.shape)))
        print("W1 gradient shape: {}".format(str(gradW1.shape)))
        print("b1 shape: {}".format(str(b1.shape)))
        print("b1 gradient shape: {}".format(str(gradb1.shape)))
    
    # ## 梯度拼起来
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    return cost, grad

    
