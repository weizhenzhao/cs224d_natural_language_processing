'''
Created on 2017年9月17日

@author: weizhen
'''
"""
import random
import numpy as np
from q3_word2vec import skipgram,word2vec_sgd_wrapper,negSamplingCostAndGradient
from q3_sgd import sgd,load_saved_params
from data_utils import StanfordSentiment
# Reset the random seed to make sure that everyone gets the same results
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

#going to train 10-dimensional vectors for this assignment
dimVectors = 10

#Context size
C=5
# Train word vectors (this could take a while!)

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors, 
                              np.zeros((nWords, dimVectors))), axis=0)
wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), 
                   wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# sanity check: cost at convergence should be around or below 10

# sum the input and output word vectors

#st, wordVectors0, state2 = load_saved_params();
wordVectors = (wordVectors0[:int(nWords),:] + wordVectors0[int(nWords):,:])

print("\n=== For autograder ===")
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print(checkVecs)
"""