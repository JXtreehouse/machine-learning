# encoding=utf-8
import tensorflow
import keras.backend as K
import jieba
import jieba.posseg as pseg
import jieba.analyse

str1 = "我来到北京清华大学"
str2 = 'python的正则表达式是好用的'
str3 = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"


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


def load_data():
    return []
