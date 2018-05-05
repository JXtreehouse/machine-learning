import time
import numpy as np
import config

import tensorflow as tf


class ChatBotModel:
    def __init__(self, forward_only, batch_size):
        print('-初始化模型')
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        print('--创建占位符')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # 右移一位去掉GO符号
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print('--创建变量')
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        #sample classes 数量要比 total classes数量大，这里的classes指vocab词汇
        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                              biases=b,
                                              inputs=logits,
                                              labels=labels,
                                              num_sampled=config.NUM_SAMPLES,
                                              num_classes=config.DEC_VOCAB)

        self.softmax_loss_function = sampled_loss
        single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])

    def _create_loss(self):
        print('--创建损失函数')

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=config.ENC_VOCAB,
                num_decoder_symbols=config.DEC_VOCAB,
                embedding_size=config.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                config.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)

            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, self.output_projection[0] + self.output_projection[1])
                                            for output in self.outputs]

        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                config.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function)

    def _create_optimizer(self):
        print('--创建优化函数')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                for bucket in range(len(config.BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(
                        tf.gradients(self.losses[bucket], trainables),
                        config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
