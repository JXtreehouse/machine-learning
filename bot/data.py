import os
import random
import re
import random
import sys
import nltk
import itertools
from collections import defaultdict
import numpy as np
import jieba as jb
import pickle

import config

regex_cn = "^[\u4E00-\u9FA5]+$"


def read_lines(filename):
    return open(filename, encoding='utf-8').read().split('\n')[:100]


def line_ids(line, lookup, maxlen):
    indices = []
    for word in line:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[config.UNK_ID])
    return indices + [config.PAD_ID] * (maxlen - len(line))


def process_raw_data():
    file_path = os.path.join(config.DATA_PATH, config.DATA_FILE)
    lines = read_lines(filename=file_path)
    freq_dist = nltk.FreqDist(itertools.chain(*lines))
    vocab = freq_dist.most_common()
    config.VOCAB_SIZE = len(vocab)
    id2word = [' '] + [config.UNK_ID] + [x[0] for x in vocab]
    word2id = dict([(w, i) for i, w in enumerate(id2word)])

    qlines = []
    alines = []

    for i in range(0,len(lines),2):
        if i + 1 == len(lines):
            break
        if re.match(regex_cn, lines[i]) and re.match(regex_cn, lines[i + 1]):
            qlines.append(lines[i])
            alines.append(lines[i + 1])

    questions = [[w for w in jb.cut(wordlist)] for wordlist in qlines]
    answers = [[w for w in jb.cut(wordlist)] for wordlist in alines]

    questions_ids = [line_ids(word, word2id, config.SENTENCE_MAX_LEN) for word in questions]
    answers_ids = [line_ids(word, word2id, config.SENTENCE_MAX_LEN) for word in answers]

    np.save(config.DATA_PATH + '/idx_q.npy', questions_ids)
    np.save(config.DATA_PATH + '/idx_a.npy', answers_ids)

    metadata = {
        'w2idx': word2id,
        'idx2w': id2word,
        'freq_dist': freq_dist
    }
    with open(config.DATA_PATH + '/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    return id2word, word2id, questions_ids, answers_ids, freq_dist

def load_data():
    # read data control dictionaries
    with open(config.DATA_PATH +'/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(config.DATA_PATH + '/idx_q.npy')
    idx_a = np.load(config.DATA_PATH + '/idx_a.npy')

    c = list(zip(idx_q, idx_a))

    random.Random().shuffle(c)

    idx_q, idx_a = zip(*c)

    return metadata, idx_q, idx_a

if __name__ == '__main__':
    process_raw_data()
