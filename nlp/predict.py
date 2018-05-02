from model import model_
from util import *
import numpy as np

n_a = 32
n_s = 64
s0 = np.zeros((1, n_s))
c0 = np.zeros((1, n_s))

m = None
Tx = 10  # 问题有多少个单词
Ty = Tx  # 回答有多少个单词

model = model_(Tx, Ty, n_a, n_s, 9097)
model.load_weights('data/model.h5')

def predict(sentence):
    list = np.array([cut2vec_input(sentence)])
    prediction = model.predict([list, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    return [i[0] for i in prediction]


print(predict('谢谢'))
