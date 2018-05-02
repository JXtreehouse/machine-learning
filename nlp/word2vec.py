# -*- coding: UTF-8 -*-
import gensim
import numpy as np
import math
from util import *

model = gensim.models.Word2Vec.load('D:\workspace\word2vec_from_weixin\word2vec\word2vec_wx')


def word2vector(word):
    try:
        return model[word]
    except KeyError:
        return np.zeros((256,))


def read_ws():
    f = open('data/words.txt', 'r', encoding="utf-8")
    str = f.read()
    words = eval(str)
    f.close()
    return words


def vec2word(vector):
    max_cos = -10000
    match_word = ''
    words = read_ws()
    for word in words:
        try:
            v = model[word]
        except KeyError:
            continue
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)


def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1 * sqrtlen2)


def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len
