import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import cv2

# 数据集路径
data_dir = 'D://dataset/mnist'

# 自动下载 MNIST 数据集
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(dtype=float, shape=[None, 784])
y_ = tf.placeholder(dtype=float, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

keep_prob = tf.placeholder(tf.float32)

#layer1
conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 6],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
conv1_biases = tf.get_variable("conv1_biases", [6], initializer=tf.constant_initializer(0.0))
conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1 , conv1_biases))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#layer2
conv2_weights = tf.get_variable("conv2_weights", [5, 5, 6, 16],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
conv2_biases = tf.get_variable("conv2_biases", [16], initializer=tf.constant_initializer(0.0))
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#layer3
fc1_weights = tf.get_variable("fc1_weights", [7*7*16, 256],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
fc1_baises = tf.get_variable("fc1_baises", [256], initializer=tf.constant_initializer(0.1))
pool2_vector = tf.reshape(pool2, [-1, 7*7*16])
fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)

# Dropout层（即按keep_prob的概率保留数据，其它丢弃），以防止过拟合
fc1_dropout = tf.nn.dropout(fc1, keep_prob)

#layer4
fc2_weights = tf.get_variable("fc2_weights", [256, 10],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))
fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

#output layer
y_conv = tf.nn.softmax(fc2)

# 加载 MNIST 模型
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./ckpt"))

    # 随机提取 MNIST 测试集的一个样本数据和标签
    test_len=len(mnist.test.images)
    test_idx=random.randint(0,test_len-1)
    x_image=mnist.test.images[test_idx]
    y=np.argmax(mnist.test.labels[test_idx])

    # 跑模型进行识别
    y_conv = tf.argmax(y_conv,1)
    pred=sess.run(y_conv,feed_dict={x:[x_image], keep_prob: 1.0})
    im = x_image.reshape(28, 28)
    cv2.imshow("pic", im)
    print('正确：',y,'，预测：',pred[0])
    cv2.waitKey(0)
