import numpy as np

# def propagate(w, b, X, Y):
#     A = sigmoid(np.dot(w.T, X) + b)
#      cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log (1-A),axis = 1, keepdims=True)
#
#
#     dw = (1 / m) * np.dot(X, (A - Y).T)
#     db = (1 / m) * np.sum(A - Y, axis=1, keepdims=True)
#
#     assert (dw.shape == w.shape)
#     assert (db.dtype == float)
#     cost = np.squeeze(cost)
#     assert (cost.shape == ())
#
#     grads = {"dw": dw, "db": db}
#
#     return grads, cost

