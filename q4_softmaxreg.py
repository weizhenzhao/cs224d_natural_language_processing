'''
Created on 2017年9月18日

@author: weizhen
'''
import numpy as np
import random
from data_utils import *
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params

def getSentenceFeature(tokens, wordVectors, sentence):
    """
        对上一步训练好的词向量
        对每一个句子中的全体词向量计算平均值作其特征值
        并试图预测所提句子中的情感层次
        超级消极，比较消极，中立，积极，非常积极
        对其分别从0到4进行编码。
        使用SGD来训练一个softmax回归机，
        并通过不断地训练/调试验证来提高回归机的泛化能力
    
        输入:
        tokens:a dictionary that maps words to their indices in the word vector list
        wordVectors: word vectors(each row) for all tokens
        sentence:a list of words in the sentence of interest
    
        输出:
        sentVector:feature vector for the sentence
    """
    sentVector = np.zeros((wordVectors.shape[1],))
    indices = [tokens[word] for word in sentence]
    sentVector = np.mean(wordVectors[indices, :], axis=0)
    return sentVector

def softmaxRegression(features, labels, weights, regularization=0.0, nopredictions=False):
    """Softmax Regression
       完成正则化的softmax回归
       输入:
       features:feature vectors,each row is a feature vector
       labels  :labels corresponding to the feature vectors
       weights :weights of the regressor
       regularization:L2 regularization constant
    
        输出:
        cost:cost of the regressor
        grad:gradient of the regressor cost with respect to its weights
        pred:label predictions of the regressor
    """
    prob = softmax(features.dot(weights))
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    """
       a vectorized implementation of 1/N * sum(cross_entropy(x_i,y_i))+1/2*|w|^2
    """
    cost = np.sum(-np.log(prob[range(N), labels])) / N
    cost += 0.5 * regularization * np.sum(weights ** 2)
    
    grad = np.array(prob)
    grad[range(N), labels] -= 1.0
    grad = features.T.dot(grad) / N
    grad += regularization * weights
    
    if N > 1:
        pred = np.argmax(prob, axis=1)
    else:
        pred = np.argmax(prob)
    
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def accuracy(y, yhat):
    """Precision for classifier"""
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization=0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights, regularization)
    return cost, grad

def sanity_check():
    """
       Run python q4_softmaxreg.py
    """
    random.seed(314159)
    np.random.seed(265)
    
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)
    
    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
    dimVectors = wordVectors.shape[1]
    
    dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
    dummy_features = np.zeros((10, dimVectors))
    dummy_labels = np.zeros((10,), dtype=np.int32)
    for i in range(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
    print("====Gradient check for softmax regression=========")
    gradcheck_naive(lambda weights:softmaxRegression(dummy_features, dummy_labels, weights, 1.0, nopredictions=True), dummy_weights)
    print("=======Results============")
    print(softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0))

if __name__ == "__main__":
    sanity_check()    
    
    

    
