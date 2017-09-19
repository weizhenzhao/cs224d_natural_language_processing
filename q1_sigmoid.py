'''
Created on 2017年9月13日

@author: weizhen
'''
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
