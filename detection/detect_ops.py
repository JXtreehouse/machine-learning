"""
Detection ops for Yolov2
"""

import tensorflow as tf
import numpy as np


def decode(detection_feat, feat_sizes=(13, 13), num_classes=80,
           anchors=None):
    """decode from the detection feature"""
    H, W = feat_sizes
    num_anchors = len(anchors)
    # 1,13,13,425 -> 1,169,5,85
    detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors,
                                                   num_classes + 5])

    # 四个值在线计算，无需反馈计算
    bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, 0:2])
    bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
    obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
    class_probs = tf.nn.softmax(detetion_results[:, :, :, 5:])

    anchors = tf.constant(anchors, dtype=tf.float32)

    height_ind = tf.range(H, dtype=tf.float32)
    width_ind = tf.range(W, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    x_offset = tf.reshape(x_offset, [1, -1, 1])  # 1,169,1
    y_offset = tf.reshape(y_offset, [1, -1, 1])  # 1,169,1

    # decode
    # x,y相对各自单元格的坐标，与yolov1一样
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    # w,h相对各自的anchor而言，与yolov1不同
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5

    # bboxes相对13x13的特征图左上右下坐标
    # 1,169,5,4
    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h,
                       bbox_x + bbox_w, bbox_y + bbox_h], axis=3)

    return bboxes, obj_probs, class_probs
