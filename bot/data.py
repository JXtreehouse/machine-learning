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
import json
import config

regex_cn = "^[\u4E00-\u9FA5]+$"
content_xml_reg = '^.*<.*>.*$'
sub_xml_reg = '<.*>'
sub_symbol_reg = "[\s+\.\!\/_,$%^*()+\"\'\:\-\=\;]+|[+——！，。？?、~@#￥%……&*（）\：\；\‘\’\“\”\、\-\=]+|[A-Za-z0-9]+|[☆机器人访客☆]"


def read_lines(filename):
    return open(filename, encoding='utf-8').readlines()[:100000]


def line_ids(line, lookup, maxlen):
    indices = []
    for word in line:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[config.UNK_ID])
    return indices


def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))


def _pad_decoder(input_, size):
    if len(input_) == size:
        return input_
    else:
        return input_ + [config.EOS_ID] + [config.PAD_ID] * (size - len(input_) - 1)


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def build_vocab(vocab, lines):
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    return vocab


def filter(line):
    if len(line) > 100 or len(line) < 5: return ''
    if re.match(content_xml_reg, line):
        line = re.sub(sub_xml_reg, '', line)
    line = re.sub(sub_symbol_reg, "", line)  # 去掉中英文符号
    line = ''.join(re.findall(r'[\u4e00-\u9fa5]', line))
    return line


def process_raw_data():
    lines = read_lines('D:\workspace\chatdata\chatdata')

    qlines = []
    alines = []

    print('loaded')

    for i in range(len(lines)):
        if lines[i] == '\n': continue
        session = json.loads(lines[i])
        for j in range(0, len(session), 2):
            if j + 1 == len(session): break
            if session[j] != '' and session[j + 1] != '':
                session[j] = filter(session[j])
                session[j + 1] = filter(session[j + 1])
                if session[j] == '\n' or session[j] == '' or session[j] == ' ' \
                        or session[j + 1] == '\n' or session[j + 1] == '' or session[j + 1] == ' ': continue
                qlines.append(session[j])
                alines.append(session[j + 1])

    print('filtered')

    questions = [[w for w in jb.cut(wordlist)] for wordlist in qlines]
    answers = [[w for w in jb.cut(wordlist)] for wordlist in alines]

    vocab = build_vocab({}, questions)
    vocab = build_vocab(vocab, answers)

    id2word = [' '] + [config.UNK_ID] + [x for x in vocab]
    word2id = dict([(w, i) for i, w in enumerate(id2word)])

    with open('config.py', 'a') as cf:
        cf.write('DEC_VOCAB = ' + str(len(id2word)) + '\n')
        cf.write('ENC_VOCAB = ' + str(len(id2word)) + '\n')

    print('wrote config')

    questions_ids = [line_ids(word, word2id, config.SENTENCE_MAX_LEN) for word in questions]
    answers_ids = [line_ids(word, word2id, config.SENTENCE_MAX_LEN) for word in answers]

    np.save(config.DATA_PATH + '/idx_q.npy', questions_ids)
    np.save(config.DATA_PATH + '/idx_a.npy', answers_ids)

    print('wrote qa')

    metadata = {
        'w2idx': word2id,
        'idx2w': id2word
    }
    with open(config.DATA_PATH + '/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print('wrote metadata')

    return id2word, word2id, questions_ids, answers_ids


def load_data():
    # read data control dictionaries
    with open(config.DATA_PATH + '/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(config.DATA_PATH + '/idx_q.npy')
    idx_a = np.load(config.DATA_PATH + '/idx_a.npy')

    c = list(zip(idx_q, idx_a))

    random.Random().shuffle(c)

    idx_q, idx_a = zip(*c)

    return metadata, idx_q, idx_a


def load_bucket_data(encode_ids, decode_ids, max_training_size=None):
    train_data_buckets = [[] for _ in config.BUCKETS]
    test_data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    for i in range(int(len(encode_ids) * (config.TRAIN_PERCENTAGE / 100))):
        encode_id = encode_ids[i]
        decode_id = decode_ids[i]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_id) <= encode_max_size and len(decode_id) <= decode_max_size:
                train_data_buckets[bucket_id].append([encode_id, decode_id])
                break

    j = i
    i = 0
    for i in range(j, len(encode_ids)):
        encode_id = encode_ids[i]
        decode_id = decode_ids[i]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_id) <= encode_max_size and len(decode_id) <= decode_max_size:
                test_data_buckets[bucket_id].append([encode_id, decode_id])
                break
    return train_data_buckets, test_data_buckets


def sentence2id(sentence, vocab):
    sentence = jb.cut(sentence)
    return [vocab[word] for word in sentence]


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_decoder(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


if __name__ == '__main__':
    process_raw_data()
