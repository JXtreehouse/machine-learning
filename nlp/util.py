# -*- coding: UTF-8 -*-
import tensorflow
import keras.backend as K
import jieba
import jieba.posseg as pseg
import jieba.analyse
import sys
from word2vec import *

sentence_size = 10


def get_train_set(input):
    i = 0
    X = np.zeros((10000, sentence_size, 256))
    Y = np.zeros((10000, sentence_size, 256))
    input_file = open(input, "r", encoding="utf-8")
    while i < 10000:
        question = input_file.readline()
        question = question.strip()
        answer = input_file.readline()
        answer = answer.strip()
        x_vec = cut2vec(question)
        y_vec = cut2vec(answer)
        x_vec = pad(x_vec)
        y_vec = pad(y_vec)
        X[i, :] = x_vec
        Y[i, :] = y_vec
        i += 1
    input_file.close()
    return X, Y


def cut2vec(sentence):
    sentence_vec_list = np.zeros((sentence_size, 256))
    seg_list = jieba.cut(sentence)
    i = 0
    for word in seg_list:
        if i == sentence_size: break
        make_dict(word)
        vec = word2vector(word)
        sentence_vec_list[i, :] = list(vec)
        i += 1
    return sentence_vec_list


def pad(vec_list):
    while sentence_size - len(vec_list) > 0:
        vec_list[len(vec_list), :] = np.zeros((256, 1))
    return vec_list


words = []
word2id = {}
id2word = {}
i = 0


def make_dict(word):
    if word not in words:
        global i
        words.append(word)
        word2id[word] = i
        id2word[i] = word
        i += 1


def save_words():
    f = open('data/word2id.txt', 'w', encoding="utf-8")
    f.write(str(word2id))
    f.close()
    f = open('data/id2word.txt', 'w', encoding="utf-8")
    f.write(str(id2word))
    f.close()


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
