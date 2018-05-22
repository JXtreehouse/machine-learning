# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
import jieba
from collections import defaultdict

trainFilePath = 'D:\\MyConfiguration\\szj46941\\Desktop\\beer标记过数据.csv'
stopWordsFilePath = 'D:\\MyConfiguration\\szj46941\\Desktop\\stopwords.txt'
stopwords = open(stopWordsFilePath).readlines()[0:-1]
stopwords = [w.strip() for w in stopwords]


def tokenization(sentences):
    results = []
    for sentence in sentences:
        words = jieba.cut(sentence)
        result = []
        for word in words:
            if word not in stopwords:
                result.append(word.lower())
        results.append(result)
    return results


def initTfIdf():
    documents = open(trainFilePath).readlines()[1:-1]
    # 1.去停顿词
    stopwords = open(stopWordsFilePath).readlines()[0:-1]
    stopwords = [w.strip() for w in stopwords]
    results = tokenization(documents)
    # 2.计算词频
    frequency = defaultdict(int)  # 构建一个字典对象
    for text in results:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text] for text in results]
    # 3.创建字典（单词与编号之间的映射）
    dictionary = corpora.Dictionary(texts)
    # 4.建立语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    index = similarities.MatrixSimilarity(corpus_tfidf)
    return index, tfidf, dictionary, documents


if __name__ == '__main__':

    index, tfidf, dictionary, documents = initTfIdf()
    input = "1664啤酒 法国原装进口啤酒 克伦堡凯旋系列白啤玫瑰红果金复古黄啤组合 6种口味组合*24瓶"
    new_doc = tokenization([input])[0]
    new_vec = dictionary.doc2bow(new_doc)
    new_vec_tfidf = tfidf[new_vec]
    sims = index[new_vec_tfidf]
    bb = sims.tolist()
    aa = sorted(bb, reverse=True)
    print('input:{}'.format(input))
    candidate = [documents[bb.index(i)] for i in aa[:10]]
    for i in range(len(candidate)):
        print('index :{}, output:{}'.format(i, candidate[i]))
