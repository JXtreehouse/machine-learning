import numpy as np
from functools import reduce
from functools import wraps


def iou_test():
    b1 = np.arange(1, 10)

    b1_xy = b1[..., :2]  # get elements from 0~1
    b1_wh = b1[..., 2:4]  # get elements from 3
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = np.arange(2, 11)

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)

    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    print(iou)


def fragment_test():
    x = np.reshape(np.arange(1, 10), (3, 3))
    print(x)
    # print(x[0, ...]) # 打印第一个坐标轴的第一个tensor
    # print(x[ ..., 0]) # 打印第二个坐标轴的第一个tensor
    # print(x[ ..., 1]) # 打印第二个坐标轴的第2个tensor
    print(x[(1), :])  # 打印第二行
    print(x[(1, 0), :])  # 打印交换后的第1,2行
    print(x[(1, 0, 2), :])  # 打印交换后的第1,2行，3行


if __name__ == '__main__':
    fragment_test()
