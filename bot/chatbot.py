from bot.model import ChatBotModel
import bot.config as config
import tensorflow as tf
import numpy as np
import random
import os


# 随机获取一个bucket
def _get_random_bucket(train_buckets_scale):
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


# 验证输入长度
def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))


# 训练一步
def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target_name = model.decoder_inputs[decoder_size].name
    input_feed[last_target_name] = np.zeros([model.batch_size], dtype=np.int32)

    if not forward_only:
        output_feed = [model.train_ops[bucket_id],
                       model.gradient_norms[bucket_id],
                       model.losses[bucket_id]]

    else:
        output_feed = [model.losses[bucket_id]]
        for step in range(decoder_size):
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None
    else:
        return None, outputs[0], outputs[1:]


def _get_buckets():
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale


def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 30
    return 100


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")


def train():
    print('-开始训练')
    # test_buckets, data_buckets, train_buckets_scale = _get_buckets()

    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id],
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)