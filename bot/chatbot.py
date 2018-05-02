
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.chinese import data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/chinese/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1] #每句话20个词
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])  #一共有多少个单词
yvocab_size = xvocab_size
emb_dim = 256 #词向量维度

import seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
# sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)

# test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
#
# input_ = test_batch_gen.__next__()[0]
# output = model.predict(sess, input_)
# print(output.shape)
#
# replies = []
# for ii, oi in zip(input_.T, output):
#     q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
#     decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
#     if decoded.count('unk') == 0:
#         if decoded not in replies:
#             print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
#             replies.append(decoded)
