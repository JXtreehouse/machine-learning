# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
import jieba
import logging
from collections import defaultdict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 文档
documents = open('C:\\Users\\admin\\Desktop\\beer标记过数据.csv').readlines()[1:-1]

# 1.分词，去除停用词
stopwords = open('C:\\Users\\admin\\Desktop\\stopwords.txt').readlines()[0:-1]
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


results = tokenization(documents)

print('-----------1----------')

# 2.计算词频
frequency = defaultdict(int)  # 构建一个字典对象
# 遍历分词后的结果集，计算每个词出现的频率
for text in results:
    for token in text:
        frequency[token] += 1


texts = [[token for token in text] for text in results]
print('-----------2----------')
print(texts)


# 3.创建字典（单词与编号之间的映射）
dictionary = corpora.Dictionary(texts)
# print(dictionary)
# Dictionary(12 unique tokens: ['time', 'computer', 'graph', 'minors', 'trees']...)
# 打印字典，key为单词，value为单词的编号
print('-----------3----------')
print(dictionary.token2id)
# {'human': 0, 'interface': 1, 'computer': 2, 'survey': 3, 'user': 4, 'system': 5, 'response': 6, 'time': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}

# 4.将要比较的文档转换为向量（词袋表示方法）
# 要比较的文档
new_doc = tokenization(["1664啤酒 法国原装进口啤酒 克伦堡凯旋系列白啤玫瑰红果金复古黄啤组合 6种口味组合*24瓶"])[0]
# 将文档分词并使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其编号，然后以稀疏向量的形式返回结果
new_vec = dictionary.doc2bow(new_doc)
print('-----------4----------')
print(new_vec)
# [[(0, 1), (2, 1)]

# 5.建立语料库
# 将每一篇文档转换为向量
corpus = [dictionary.doc2bow(text) for text in texts]
print('-----------5----------')
print(corpus)
# [[[(0, 1), (1, 1), (2, 1)], [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(1, 1), (4, 1), (5, 1), (8, 1)], [(0, 1), (5, 2), (8, 1)], [(4, 1), (6, 1), (7, 1)], [(9, 1)], [(9, 1), (10, 1)], [(9, 1), (10, 1), (11, 1)], [(3, 1), (10, 1), (11, 1)]]

# 6.初始化模型
# 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
tfidf = models.TfidfModel(corpus)
# 测试
test_doc_bow = [(0, 1), (1, 1)]
print('-----------6----------')
print(tfidf[test_doc_bow])
# [(0, 0.7071067811865476), (1, 0.7071067811865476)]

print('-----------7----------')
# 将整个语料库转为tfidf表示方法
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

# 7.创建索引
index = similarities.MatrixSimilarity(corpus_tfidf)

print('-----------8----------')
# 8.相似度计算
new_vec_tfidf = tfidf[new_vec]  # 将要比较文档转换为tfidf表示方法
print(new_vec_tfidf)
# [(0, 0.7071067811865476), (2, 0.7071067811865476)]
print('-----------9----------')
# 计算要比较的文档与语料库中每篇文档的相似度
sims = index[new_vec_tfidf]
print(sims)
bb = sims.tolist()
aa = sorted(bb,reverse = True)
print([bb.index(i)+2 for i in aa[:10]])
# [ 0.81649655  0.31412902  0.          0.34777319  0.          0.          0.
#  0.          0.        ]
