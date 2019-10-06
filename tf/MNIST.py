import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据集路径
data_dir = 'D://dataset/mnist'

# 自动下载 MNIST 数据集
mnist = input_data.read_data_sets(data_dir, one_hot=True)

train_xdata = mnist.train.images
test_xdata = mnist.test.images

train_labels = mnist.train.labels
test_labels = mnist.test.labels

step_cnt = 3000
batch_size = 100
learning_rate = 0.001

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

# 定义交叉熵损失函数，越相似交叉熵越小
# y_ 为真实标签
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# 选择优化器，使优化器最小化损失函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 返回模型预测的最大概率的结果，并与真实值作比较
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 用平均值来统计测试准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练模型
saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(step_cnt):
        batch = mnist.train.next_batch(batch_size)
        if step % 100 == 0:
            # 每迭代100步进行一次评估，输出结果，保存模型，便于及时了解模型训练进展
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (step, train_accuracy))
            saver.save(sess,'./ckpt/mnist.ckpt',global_step=step)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

    # 使用测试数据测试准确率
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))