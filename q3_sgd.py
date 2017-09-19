'''
Created on Sep 12, 2017

@author: p0079482

实现随机梯度下降
随机梯度下降每1000轮，就保存一下现在训练得到的参数
'''
SAVE_PARAMS_EVERY = 1000

import glob
import os.path as op
import pickle  as pickle
import sys
import random

def load_saved_params():
    """
        载入之前的参数以免从头开始训练
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if(iter > st):
            st = iter
    
    if st > 0:
        with open("saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10, ANNEAL_EVERY=20000):
    """随机梯度下降
    输入:
    f:需要最优化的函数
    x0:SGD的初始值
    step:SGD的步长
    iterations：总的迭代次数
    postprocessing:参数后处理(比如word2vec里需要对词向量做归一化处理)
    PRINT_EVERY:指明多少次迭代以后输出一下状态
  输出:
    x:SGD完成后的输出参数
    """
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x:x
    
    expcost = None
    
    for iter in range(start_iter + 1, iterations + 1):
        cost, grad = f(x)
        x = x - step * grad
        x = postprocessing(x)
        
        if iter % PRINT_EVERY == 0:
            print("iter#{},cost={}".format(iter, cost))
            sys.stdout.flush()
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    return x

    
