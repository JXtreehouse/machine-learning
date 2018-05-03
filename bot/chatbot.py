# In[1]:

import tensorflow as tf
import numpy as np
import random

tf.device('/gpu:0')
# preprocessed data
from datasets.chinese import data
import data_utils

import seq2seq_wrapper

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/chinese/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]  # 每句话20个词
yseq_len = trainY.shape[-1]
batch_size = 1024
xvocab_size = len(metadata['idx2w'])  # 一共有多少个单词
yvocab_size = xvocab_size
emb_dim = 512


# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path='ckpt/twitter/',
                                emb_dim=emb_dim,
                                num_layers=3,
                                epochs=301,
                                lr=0.00005
                                )

# In[8]:


train = True

if train:
    for i in range(10):
        c = list(zip(idx_q, idx_a))

        random.Random().shuffle(c)

        idx_q, idx_a = zip(*c)

        idx_q= np.array(idx_q)
        idx_a= np.array(idx_a)

        (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

        val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
        train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
        sess = model.restore_last_session()
        sess = model.train(train_batch_gen, val_batch_gen,sess)
else:
    sess = model.restore_last_session()
    test_batch_gen = data_utils.rand_batch_gen(testX, testY, 128)

    input_ = test_batch_gen.__next__()[0]
    output = model.predict(sess, input_)
    print(output.shape)

    replies = []
    for ii, oi in zip(input_.T, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)
