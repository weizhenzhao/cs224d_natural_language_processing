'''
Created on Sep 11, 2017

@author: p0079482
'''
import random
import numpy as np
def gradcheck_naive(f,x):
    """ 对一个函数f求梯度检验
        -f 输入x,然后输出loss和梯度的函数
        -x 就是输入
     """
     
    rndstate = random.getstate() 
      #获取当前随机数的环境状态，以便下次再使用
    random.setstate(rndstate)
    fx,grad=f(x)
    h=1e-4
     
     #遍历x的每一维
     #单个数组的迭代
     #nditer将输入数组视为只读对象。要修改数组元素，必须指定读写（ read-write）或只写（write-only）模式
     
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
         ix=it.multi_index
         old_val=x[ix]
         x[ix]=old_val-h
         random.setstate(rndstate)
         (fxh1,_)=f(x)
         
         x[ix]=old_val+h
         random.setstate(rndstate)
         (fxh2,_)=f(x)
         
         numgrad = (fxh2-fxh1)/(2*h)
         x[ix] = old_val
         
         #对比梯度
         reldiff = abs(numgrad-grad[ix])/max(1,abs(numgrad),abs(grad[ix]))
         if reldiff>1e-5:
             print("Gradient check failed.")
             print("First gradient error found at index %s"%str(ix))
             print("Your gradient: %f \t Numerical gradient: %f"%(grad[ix],numgrad))
             return
         it.iternext()  #Step to next dimension
    print("Gradient check passed!")
      
    
         
         
         