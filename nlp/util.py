# -*- coding: UTF-8 -*-
import tensorflow
import keras.backend as K
import jieba
import jieba.posseg as pseg
import jieba.analyse
import sys
import numpy as np

sentence_size = 10
thresh_holder = 1000
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1
# 结尾标记
EOS_ID = 2
UNK_ID = 3
START_ID = 4
word2id_dict = {}
id2word_dict = {}
oh_len = 0


def word2id(word):
    if not isinstance(word, str):
        print("Exception: error word not unicode")
        sys.exit(1)
    if word in word2id_dict:
        return word2id_dict[word]
    else:
        return None


def word2oh(word):
    id = word2id(word)
    vec = np.zeros((oh_len,))
    vec[id] = 1
    return vec


def id2word(id):
    id = int(id)
    if id in id2word_dict:
        return id2word_dict[id]
    else:
        return None

def get_train_set(input, min_freq):
    i = 0

    words_count = {}

    input_file = open(input, "r", encoding="utf-8")
    while i < thresh_holder:
        question = input_file.readline()
        question = question.strip()
        answer = input_file.readline()
        answer = answer.strip()
        seg_list = jieba.cut(question)
        for str in seg_list:
            if str in words_count:
                words_count[str] = words_count[str] + 1
            else:
                words_count[str] = 1

        seg_list = jieba.cut(answer)
        for str in seg_list:
            if str in words_count:
                words_count[str] = words_count[str] + 1
            else:
                words_count[str] = 1

        i += 1
    input_file.close()

    sorted_list = [[v[1], v[0]] for v in words_count.items()]
    sorted_list.sort(reverse=True)
    for index, item in enumerate(sorted_list):
        word = item[1]
        if item[0] < min_freq:
            break
        word2id_dict[word] = START_ID + index
        id2word_dict[START_ID + index] = word

    global oh_len

    word2id_dict['PAD_ID'] = 0
    word2id_dict['GO_ID'] = 1
    word2id_dict['EOS_ID'] = 2
    word2id_dict['UNK_ID'] = 3
    id2word_dict[0] = 'PAD_ID'
    id2word_dict[1] = 'GO_ID'
    id2word_dict[2] = 'EOS_ID'
    id2word_dict[3] = 'UNK_ID'

    oh_len = len(word2id_dict)

    X = np.zeros((thresh_holder, sentence_size, oh_len))
    Y = np.zeros((thresh_holder, sentence_size, oh_len))

    i = 0
    input_file = open(input, "r", encoding="utf-8")
    while i < thresh_holder:
        question = input_file.readline()
        question = question.strip()
        answer = input_file.readline()
        answer = answer.strip()
        x_vec = cut2vec_input(question)
        y_vec = cut2vec_output(answer)
        X[i, :] = x_vec
        Y[i, :] = y_vec
        i += 1
    input_file.close()

    return X, Y


def cut2vec_input(sentence):
    sentence_vec_list = np.zeros((sentence_size, oh_len))
    seg_list = jieba.cut(sentence)
    i = 0
    for word in seg_list:
        if i == sentence_size: break
        vec = word2oh(word)
        sentence_vec_list[i, :] = list(vec)
        i += 1
    for j in range(sentence_vec_list.shape[0] - i - 1):
        sentence_vec_list[j, :] = list(word2oh('PAD_ID'))

    return sentence_vec_list


def cut2vec_output(sentence):
    sentence_vec_list = np.zeros((sentence_size, oh_len))
    seg_list = jieba.cut(sentence)
    i = 0

    sentence_vec_list[i, :] = list(word2oh('GO_ID'))

    i= 1

    for word in seg_list:
        if i == sentence_size: break
        vec = word2oh(word)
        sentence_vec_list[i, :] = list(vec)
        i += 1

    sentence_vec_list[i-1, :] = word2oh('EOS_ID')

    for j in range(sentence_vec_list.shape[0] - i - 2):
            sentence_vec_list[j, :] = list(word2oh('PAD_ID'))

    return sentence_vec_list


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
