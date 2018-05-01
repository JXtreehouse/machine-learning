# encoding=utf-8
import sys
import jieba
from word2vec import *


def get_train_set():
    X = []
    Y = []
    with open('./data/question.txt', 'r',encoding='UTF-8') as question_file:
        with open('./data/answer.txt', 'r',encoding='UTF-8') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    question = question.strip()
                    answer = answer.strip()
                    x_vec = cut2vec(question)
                    y_vec = cut2vec(answer)
                    X.append(x_vec)
                    Y.append(y_vec)
                else:
                    break
    return X, Y


def cut2vec(sentence):
    sentence_vec_list = []
    seg_list = jieba.cut(sentence)
    for word in seg_list:
        vec = word2vector(word)
        sentence_vec_list.append(vec)
    return sentence_vec_list

# X,Y = get_train_set()
# print(X)
# print(Y)
