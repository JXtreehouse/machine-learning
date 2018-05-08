from bot.model import ChatBotModel
import bot.config as config
import tensorflow as tf
import numpy as np
import random
import os
import data
import time
import sys



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


# 随机获取一个bucket
def _get_random_bucket(train_buckets_scale):
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def _get_buckets():
    metadata, idx_q, idx_a = data.load_data()
    train_data_buckets, test_data_buckets = data.load_bucket_data(idx_q, idx_a)
    train_bucket_sizes = [len(train_data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    train_buckets_scale = [sum(train_bucket_sizes[i:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_data_buckets, train_data_buckets, train_buckets_scale, metadata


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


def _eval_test_set(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_buckets[bucket_id],
                                                                       bucket_id,
                                                                       batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))


def _construct_response(output_logits, metadata):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    outputs = [int(np.argmax(logit)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(id2w(metadata['idx2w'], output)) for output in outputs])


def id2w(idx2w, id):
    return idx2w[id]


def train():

    print('-开始训练')
    test_data_buckets, train_data_buckets, train_buckets_scale, _ = _get_buckets()

    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        learning_rate = model.learning_rate.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(train_data_buckets[bucket_id],
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            print('Iter {}:  loss{}, learning rate{}, time {}'.format(iteration, step_loss, learning_rate,
                                                                      time.time() - start))
            total_loss += step_loss
            iteration += 1
            if iteration % skip_step == 0:
                print(
                    'Saved at iter {}: loss {}, time {}'.format(iteration, total_loss / skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'))
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_data_buckets)
                    start = time.time()
                sys.stdout.flush()


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])


def chat():
    """ in test mode, we don't to create the backward path
    """
    test_data_buckets, train_data_buckets, train_buckets_scale, metadata = _get_buckets()

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        bucket_id = _get_random_bucket(train_buckets_scale)
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_data_buckets[bucket_id],
                                                                       bucket_id,
                                                                       batch_size=20)

        # Get output logits for the sentence.
        _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                       decoder_masks, bucket_id, True)

        for logit in output_logits:

            response = _construct_response(logit, metadata)
            print(response)


train_mode = True

if __name__ == '__main__':
    if train_mode:
        train()
    else:
        chat()
