'''
Created on Sep 11, 2017

@author: p0079482
'''
def sigmoid_graf(f):
    """
        计算Sigmoid的梯度
    """
    f = f * (1 - f)
    
    return f
