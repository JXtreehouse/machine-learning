# -*- coding: UTF-8 -*-
import numpy as np

def normalize(vec):
    return np.where(vec > 0, vec, -1*vec)


a = np.random.rand(10,10)
a = -1*a
print(a)
print(normalize(a))

