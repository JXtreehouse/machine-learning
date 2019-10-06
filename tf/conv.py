import tensorflow as tf

# input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，
# 其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3(rgb)。（也可以用其它值，但是具体含义不是很理解）
# filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
# 其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
# strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
# padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
# use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

# case 1
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 1*1 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))
op1 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 2
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 2*2 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([2, 2, 5, 1]))
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 3
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 1*1 的feature map (不考虑边界)
# 1张图最后输出就是一个 shape为[1,1,1,1] 的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# case 4
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map (不考虑边界)
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# case 5
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 5*5 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,5,5,1] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 6
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,1,1,1]最后得到一个 5*5 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,5,5,7] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 7
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后得到7个 3*3 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,3,3,7] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

# case 8
# 输入是10 张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后每张图得到7个 3*3 的feature map (考虑边界)
# 10张图最后输出就是一个 shape为[10,3,3,7] 的张量
input = tf.Variable(tf.random_normal([10, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print('*' * 20 + ' op1 ' + '*' * 20)
    print(sess.run(op1))
    print('*' * 20 + ' op2 ' + '*' * 20)
    print(sess.run(op2))
    print('*' * 20 + ' op3 ' + '*' * 20)
    print(sess.run(op3))
    print('*' * 20 + ' op4 ' + '*' * 20)
    print(sess.run(op4))
    print('*' * 20 + ' op5 ' + '*' * 20)
    print(sess.run(op5))
    print('*' * 20 + ' op6 ' + '*' * 20)
    print(sess.run(op6))
    print('*' * 20 + ' op7 ' + '*' * 20)
    print(sess.run(op7))
    print('*' * 20 + ' op8 ' + '*' * 20)
    print(sess.run(op8))
