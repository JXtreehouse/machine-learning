import gensim
import numpy as np

model = gensim.models.Word2Vec.load('D:\workspace\word2vec_from_weixin\word2vec\word2vec_wx')

np.random.seed(1)

def word2vector(word):
    try:
        return model[word]
    except KeyError:
        return np.random.rand(256,1)

