'''
Created on Sep 11, 2017

@author: p0079482
'''
"""实现word2vec模型，并使用随机梯度下降方法SGD训练属于自己的词向量
      首先写一个辅助函数对矩阵中的每一行进行归一化，
      同样在这个文件中完成对softmax,
      负采样损失函数
      以及梯度计算函数的实现
      然后完成面向skip-gram的梯度损失函数
"""
import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid_graf
from q1_sigmoid import sigmoid

def normalizeRows(x):
    """
       行归一化函数
    """
    N=x.shape[0]
    x/=np.sqrt(np.sum(x**2,axis=1)).reshape((N,1))+1e-30
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows")
    x = normalizeRows(np.array([[3.0,4.0],[1,2]]))
    """结果应该是[[0.6,0.8],[0.4472136,0.89442719]]"""
    print(x)
    assert(np.amax(np.fabs(x-np.array([[0.6,0.8],[0.4472136,0.89442719]])))<= 1e-6)
    print(" ")

def softmaxCostAndGradient(predicted,target,outputVectors,dataset):
    """word2vec的Softmax损失函数"""
    
    """
       输入:
       predicted：预测词向量的numpy数组
       target：目标词的下标
       outputVectors：所有token的"output"向量(行形式)
       dataset：用来做负例采样的，这里其实没用着
    
        输出:
        cost:输出的互熵损失
        gradPred:the gradient with respect to the predicted word vector
        grad:    the gradient with respect to all the other word vectors
    """
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
    delta = probabilities
    delta[target] -= 1
    N = delta.shape[0]
    D = predicted.shape[0]
    grad = delta.reshape((N,1))*predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()
    
    return cost,gradPred,grad

def negSamplingCostAndGradient(predicted,target,outputVectors,dataset,K=10):
    """
        Word2vec模型负例采样后的损失函数和梯度
    """
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices+=[newidx]
    
    labels = np.array([1]+[-1 for k in range(K)])
    vecs = outputVectors[indices,:]
    
    t=sigmoid(vecs.dot(predicted)*labels)
    cost = -np.sum(np.log(t))
    
    delta = labels*(t-1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape((1,predicted.shape[0])))
    for k in range(K+1):
        grad[indices[k]]+=gradtemp[k,:]
    
    t = sigmoid(predicted.dot(outputVectors[target,:]))
    cost = -np.log(t)
    delta = t - 1
    
    gradPred+=delta*outputVectors[target,:]
    grad[target,:] += delta*predicted
    
    for k in range(K):
        idx = dataset.sampleTokenIdx()
        
        t = sigmoid(-predicted.dot(outputVectors[idx,:]))
        cost += -np.log(t)
        delta = 1-t
        
        gradPred += delta*outputVectors[idx,:]
        grad[idx,:]+=delta*predicted
    
    return cost,gradPred,grad

def skipgram(currentWord,C,contextWords,tokens,inputVectors,outputVectors,
             dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    """
        Skip-gram model in word2vec
        
        skip-gram模型的实现
        输入:
        currentWord:当前中心词所对应的串
        C:上下文大小(词窗大小)
        contextWords:最多2*C个词
        tokens:对应词向量中词下标的字典
        inputVectors:"input" word vectors (as rows) for all tokens
        outputVectors: "output" word vectors (as rows) for all tokens
        word2vecCostAndGradient: the cost and gradient function for a prediction vector
                                 given the target word vectors,
                                 could be one of the two cost functions you implemented
                                 above
                                 
        输出:
        cost:skip-gram模型算得的损失值
        grad:词向量对应的梯度
    """
    currentI=tokens[currentWord]
    predicted = inputVectors[currentI,:]
    
    cost=0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        cc,gp,gg = word2vecCostAndGradient(predicted,idx,outputVectors,dataset)
        cost+=cc
        gradOut += gg
        gradIn[currentI,:]+=gp
    return cost,gradIn,gradOut

def word2vec_sgd_wrapper(word2vecModel,tokens,wordVectors,dataset,C,word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost =0.0
    grad = np.zeros(wordVectors.shape)
    N=wordVectors.shape[0]
    splitIndex=int(N/2)
    inputVectors = wordVectors[:splitIndex,:]
    outputVectors = wordVectors[splitIndex:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword,context=dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom =1
        else:
            denom =1
        
        c,gin,gout = word2vecModel(centerword,C1,context,tokens,inputVectors,outputVectors,dataset,word2vecCostAndGradient)
        cost+=c/batchsize/denom
        grad[:splitIndex,:]+=gin/batchsize/denom
        grad[splitIndex:,:]+=gout/batchsize/denom
     
    return cost, grad

def test_word2vec():
    """
         Interface to the dataset for negative sampling
    """
 
    dataset = type('dummy',(),{})()
    def dummySampleTokenIdx():
        return random.randint(0,4)
     
    def getRandomContext(C):
        tokens = ["a","b","c","d","e"]
        return tokens[random.randint(0,4)],[tokens[random.randint(0,4)] for i in range(2*C)]
     
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
     
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0),("b",1),("c",2),("d",3),("e",4)])
    print("===== Gradient check for skip-gram ======")
    #gradcheck_naive(lambda vec:word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5),dummy_vectors)
    gradcheck_naive(lambda vec:word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5,negSamplingCostAndGradient),dummy_vectors)
    print("\n===== Gradient check for CBOW")
    #gradcheck_naive(lambda vec:word2vec_sgd_wrapper(cbow,dummy_tokens,vec,dataset,5),dummy_vectors)
    #gradcheck_naive(lambda vec:word2vec_sgd_wrapper(cbow,dummy_tokens,vec,dataset,5,negSamplingCostAndGradient),dummy_vectors)
     
    print("\n===== Results ===")
    print(skipgram("c",3,["a","b","e","d","b","c"],dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset))
    #print(skipgram("c",1,["a","b"],dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset,negSamplingCostAndGradient))
    #print(cbow("a",2,["a","b","c","a"],dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset))
    #print(cbow("a",2,["a","b","a","c"],dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset,negSamplingCostAndGradient))
     
#if __name__=="__main__":
#    test_normalize_rows()
#    test_word2vec()
    
        