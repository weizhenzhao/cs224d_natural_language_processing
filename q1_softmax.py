'''
Created on Sep 11, 2017

@author: p0079482
'''
import numpy as np

def softmax(x):
    """
        Softmax函数
    """
    #assert len(x.shape) > 1, "Softmax的得分向量要求维度高于1"
    #x -= np.max(x, axis=1, keepdims=True)
    #x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    x -= np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    
    return x
