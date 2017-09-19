'''
Created on 2017年9月19日

@author: weizhen
'''
import numpy as np
import matplotlib.pyplot as plt

from data_utils import *
from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper
from data_utils import StanfordSentiment
"""
完成超参数的实现代码，从而获取最佳的惩罚因子
"""
# 尝试不同的正则化系数，选取最好的
REGULARIZATION = [0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
# 载入数据集
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# 载入预训练好的词向量
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
dimVectors = wordVectors.shape[1]

# 载入训练集
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
# 准备好训练的特征
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# 尝试不同的正则化系数
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization)
    
    # batch optimization
    weights = sgd(lambda weights:softmax_wrapper(trainFeatures, trainLabels, weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)
    
    # 训练集上测效果
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%):%f" % trainAccuracy)
    
    # dev集合上看效果
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%):%f" % devAccuracy)
    
    # 保存结果权重
    results.append({
        "reg":regularization,
        "weights":weights,
        "train":trainAccuracy,
        "dev":devAccuracy
    })
# 输出准确率
print("  ")
print("===Recap===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (result["reg"], result["train"], result["dev"]))

print("   ")

best_dev = 0
for result in results:
    if result["dev"] > best_dev:
        best_dev = result["dev"]
        BEST_REGULARIZATION = result["reg"]
        BEST_WEIGHTS = result["weights"]

# Test your findings on the test set
testset = dataset.getTrainSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value:%E" % BEST_REGULARIZATION)
print("Test accuracy (%%):%f" % accuracy(testLabels, pred))

# 画出正则化和准确率的关系
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("q4_reg_v_acc.png")
plt.show()





    
    
