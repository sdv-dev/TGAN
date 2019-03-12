#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

"""TGAN_synthesizer."""


import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorpack import (
    BatchData, BatchNorm, Dropout, FullyConnected, InputDesc, ModelSaver, PredictConfig,
    QueueInput, SaverRestore, SimpleDatasetPredictor, get_model_loader, logger)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt

from tgan.GAN import GANModelDesc, GANTrainer, RandomZData
from tgan.np_data_flow import NpDataFlow

tunable_variables = {
    "--batch_size": [50, 100, 200],
    "--z_dim": [50, 100, 200, 400],
    "--num_gen_rnn": [100, 200, 300, 400, 500, 600],
    "--num_gen_feature": [100, 200, 300, 400, 500, 600],
    "--num_dis_layers": [1, 2, 3, 4, 5],
    "--num_dis_hidden": [100, 200, 300, 400, 500],
    "--learning_rate": [0.0002, 0.0005, 0.001],
    "--noise": [0.05, 0.1, 0.2, 0.3]
}


class Model(GANModelDesc):
    """Main model for TGAN.

    Args:
        None

    Attributes:

    """

    def _get_inputs(self):
        """Return metadata about entry data.

        Returns:
            list[tensorpack.InputDesc]

        """
        inputs = []
        for col_id, col_info in enumerate(opt.DATA_INFO['details']):
            if col_info['type'] == 'value':
                gaussian_components = col_info['n']
                inputs.append(
                    InputDesc(tf.float32, (opt.batch_size, 1), 'input%02dvalue' % col_id))

                inputs.append(
                    InputDesc(
                        tf.float32,
                        (opt.batch_size, gaussian_components),
                        'input%02dcluster' % col_id
                    )
                )

            elif col_info['type'] == 'category':
                inputs.append(InputDesc(tf.int32, (opt.batch_size, 1), 'input%02d' % col_id))

            else:
                assert 0

        return inputs

    def generator(self, z):
        r"""Build generator graph.

        We generate a numerical variable in 2 steps. We first generate the value scalar
        :math:`v_i`, then generate the cluster vector :math:`u_i`. We generate categorical
        feature in 1 step as a probability distribution over all possible labels.

        The output and hidden state size of LSTM is :math:`n_h`. The input to the LSTM in each
        step :math:`t` is the random variable :math:`z`, the previous hidden vector :math:`f_{t−1}`
        or an embedding vector :math:`f^{\prime}_{t−1}` depending on the type of previous output,
        and the weighted context vector :math:`a_{t−1}`. The random variable :math:`z` has
        :math:`n_z` dimensions.
        Each dimension is sampled from :math:`\mathcal{N}(0, 1)`. The attention-based context
        vector at is a weighted average over all the previous LSTM outputs :math:`h_{1:t}`.
        So :math:`a_t` is a :math:`n_h`-dimensional vector.
        We learn a attention weight vector :math:`α_t \in \mathbb{R}^t` and compute context as

        .. math::
            a_t = \sum_{k=1}^{t} \frac{\textrm{exp}  {\alpha}_{t, j}}
                {\sum_{j} \textrm{exp}  \alpha_{t,j}} h_k.

        We set :math: `a_0` = 0. The output of LSTM is :math:`h_t` and we project the output to
        a hidden vector :math:`f_t = \textrm{tanh}(W_h h_t)`, where :math:`W_h` is a learned
        parameter in the network. The size of :math:`f_t` is :math:`n_f` .
        We further convert the hidden vector to an output variable.

        * If the output is the value part of a continuous variable, we compute the output as
          :math:`v_i = \textrm{tanh}(W_t f_t)`. The hidden vector for :math:`t + 1` step is
          :math:`f_t`.

        * If the output is the cluster part of a continuous variable, we compute the output as
          :math:`u_i = \textrm{softmax}(W_t f_t)`. The feature vector for :math:`t + 1` step is
          :math:`f_t`.

        * If the output is a discrete variable, we compute the output as
          :math:`d_i = \textrm{softmax}(W_t f_t)`. The hidden vector for :math:`t + 1` step is
          :math:`f^{\prime}_{t} = E_i [arg_k \hspace{0.25em} \textrm{max} \hspace{0.25em} d_i ]`,
          where :math:`E \in R^{|D_i|×n_f}` is an embedding matrix for discrete variable
          :math:`D_i`.

        * :math:`f_0` is a special vector :math:`\texttt{<GO>}` and we learn it during the
          training.

        Args:
            z:

        Returns:
            list[]

        Raises:


        """
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(opt.num_gen_rnn)

            state = cell.zero_state(opt.batch_size, dtype='float32')
            attention = tf.zeros(shape=(opt.batch_size, opt.num_gen_rnn), dtype='float32')
            input = tf.get_variable(name='go', shape=(1, opt.num_gen_feature))  # <GO>
            input = tf.tile(input, [opt.batch_size, 1])
            input = tf.concat([input, z], axis=1)

            ptr = 0
            outputs = []
            states = []
            for col_id, col_info in enumerate(opt.DATA_INFO['details']):
                if col_info['type'] == 'value':
                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])

                    gaussian_components = col_info['n']
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, opt.num_gen_feature, nl=tf.tanh)
                        outputs.append(FullyConnected('FC2', h, 1, nl=tf.tanh))
                        input = tf.concat([h, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, dim=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, opt.num_gen_feature, nl=tf.tanh)
                        w = FullyConnected('FC2', h, gaussian_components, nl=tf.nn.softmax)
                        outputs.append(w)
                        input = FullyConnected('FC3', w, opt.num_gen_feature, nl=tf.identity)
                        input = tf.concat([input, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, dim=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                elif col_info['type'] == 'category':
                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, opt.num_gen_feature, nl=tf.tanh)
                        w = FullyConnected('FC2', h, col_info['n'], nl=tf.nn.softmax)
                        outputs.append(w)
                        one_hot = tf.one_hot(tf.argmax(w, axis=1), col_info['n'])
                        input = FullyConnected('FC3', one_hot, opt.num_gen_feature, nl=tf.identity)
                        input = tf.concat([input, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, dim=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                else:
                    assert 0

        return outputs

    @auto_reuse_variable_scope
    def discriminator(self, vecs):
        r"""Build discriminator.

        We use a :math:`l`-layer fully connected neural network as the discriminator.
        We concatenate :math:`v_{1:n_c}` , :math:`u_{1:n_c}` and :math:`d_{1:n_d}` together as the
        input. We compute the internal layers as

        .. math::
            f^{(D)}_{1} = \textrm{LeakyReLU}(\textrm{BN}(W^{(D)}_{1}(v_{1:n_c} \oplus u_{1:n_c}
                \oplus d_{1:n_d})

            f^{(D)}_{1} = \textrm{LeakyReLU}(\textrm{BN}(W^{(D)}_{i}(f^{(D)}_{i−1} \oplus
                \textrm{diversity}(f^{(D)}_{i−1})))), i = 2:l

        where :math:`\oplus` is the concatenation operation. :math:`\textrm{diversity}(·)` is the
        mini-batch discrimination vector [42]. Each dimension of the diversity vector is the total
        distance between one sample and all other samples in the mini-batch using some learned
        distance metric. :math:`\textrm{BN}(·)` is batch normalization, and
        :math:`\textrm{LeakyReLU}(·)` is the leaky reflect linear activation function. We further
        compute the output of discriminator as :math:`W^{(D)}(f^{(D)}_{l} \oplus \textrm{diversity}
        (f^{(D)}_{l}))` which is a scalar.

        Args:
            vecs()

        Returns:
            tensorpack.FullyConected

        """
        def batch_diversity(l, n_kernel=10, kernel_dim=10):
            M = FullyConnected('fc_diversity', l, n_kernel * kernel_dim, nl=tf.identity)
            M = tf.reshape(M, [-1, n_kernel, kernel_dim])
            M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
            M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
            diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
            diversity = tf.reduce_sum(diff, axis=0)
            return diversity

        """ return a (b, 1) logits"""
        logits = tf.concat(vecs, axis=1)
        for i in range(opt.num_dis_layers):
            with tf.variable_scope('dis_fc{}'.format(i)):
                if i == 0:
                    logits = FullyConnected(
                        'fc', logits, opt.num_dis_hidden, nl=tf.identity,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )

                else:
                    logits = FullyConnected('fc', logits, opt.num_dis_hidden, nl=tf.identity)

                logits = tf.concat([logits, batch_diversity(logits)], axis=1)
                logits = BatchNorm('bn', logits, center=True, scale=False)
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return FullyConnected('dis_fc_top', logits, 1, nl=tf.identity)

    def _build_graph(self, inputs):
        z = tf.random_normal(
            [opt.batch_size, opt.z_dim], name='z_train')

        z = tf.placeholder_with_default(z, [None, opt.z_dim], name='z')

        with tf.variable_scope('gen'):
            vecs_gen = self.generator(z)

            vecs_denorm = []
            ptr = 0
            for col_id, col_info in enumerate(opt.DATA_INFO['details']):
                if col_info['type'] == 'category':
                    t = tf.argmax(vecs_gen[ptr], axis=1)
                    t = tf.cast(tf.reshape(t, [-1, 1]), 'float32')
                    vecs_denorm.append(t)
                    ptr += 1

                elif col_info['type'] == 'value':
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1

                else:
                    assert 0

        vecs_pos = []
        ptr = 0
        for col_id, col_info in enumerate(opt.DATA_INFO['details']):
            if col_info['type'] == 'category':
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])
                noise_input = one_hot

                if opt.sample == 0:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=opt.noise)
                    noise_input = (one_hot + noise) / tf.reduce_sum(
                        one_hot + noise, keep_dims=True, axis=1)

                vecs_pos.append(noise_input)
                ptr += 1

            elif col_info['type'] == 'value':
                vecs_pos.append(inputs[ptr])
                ptr += 1
                vecs_pos.append(inputs[ptr])
                ptr += 1

            else:
                assert 0

        def compute_kl(real, pred):
            return tf.reduce_sum((tf.log(pred + 1e-4) - tf.log(real + 1e-4)) * pred)

        KL = 0.
        ptr = 0
        if opt.sample == 0:
            for col_id, col_info in enumerate(opt.DATA_INFO['details']):
                if col_info['type'] == 'category':
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)

                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += compute_kl(real, dist)
                    ptr += 1

                elif col_info['type'] == 'value':
                    ptr += 1
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)
                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += compute_kl(real, dist)

                    ptr += 1

                else:
                    assert 0

        with tf.variable_scope('discrim'):
            discrim_pos = self.discriminator(vecs_pos)
            discrim_neg = self.discriminator(vecs_gen)

        self.build_losses(discrim_pos, discrim_neg, extra_g=KL, l2_norm=opt.l2norm)
        self.collect_variables()

    def _get_optimizer(self):
        if opt.optimizer == 'AdamOptimizer':
            return tf.train.AdamOptimizer(opt.learning_rate, 0.5)

        elif opt.optimizer == 'AdadeltaOptimizer':
            return tf.train.AdadeltaOptimizer(opt.learning_rate, 0.95)

        else:
            return tf.train.GradientDescentOptimizer(opt.learning_rate)


def get_data(datafile):
    """Return a valid InputSource from a numpy.array file."""
    ds = NpDataFlow(datafile, shuffle=True)
    opt.distribution = ds.distribution
    return BatchData(ds, opt.batch_size)


def sample(n, model, model_path, output_name='gen/gen', output_filename=None):
    """Generate samples from model."""
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])

    pred = SimpleDatasetPredictor(
        pred, RandomZData((opt.batch_size, opt.z_dim)))

    max_iters = n // opt.batch_size
    if output_filename is None:
        output_filename = opt.exp_name if opt.exp_name else 'generate'
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        output_filename += '_{}'.format(timestamp)

    results = []
    for idx, o in enumerate(pred.get_result()):
        results.append(o[0])
        if idx + 1 == max_iters:
            break

    results = np.concatenate(results, axis=0)

    ptr = 0
    features = {}
    for col_id, col_info in enumerate(opt.DATA_INFO['details']):
        if col_info['type'] == 'category':
            features['f%02d' % col_id] = results[:, ptr:ptr + 1]
            ptr += 1

        elif col_info['type'] == 'value':
            gaussian_components = col_info['n']
            val = results[:, ptr:ptr + 1]
            ptr += 1
            pro = results[:, ptr:ptr + gaussian_components]
            ptr += gaussian_components
            features['f%02d' % col_id] = np.concatenate([val, pro], axis=1)

        else:
            assert 0

    np.savez(output_filename, info=json.dumps(opt.DATA_INFO), **features)


def get_args():
    """CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', type=int, default=0,
                        help='the number of samples in the synthetic output.')
    parser.add_argument('--data', help='a npz file')
    parser.add_argument('--output', type=str)
    parser.add_argument('--exp_name', type=str, default=None)

    # parameters for model tuning.
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)

    parser.add_argument('--num_gen_rnn', type=int, default=400)
    parser.add_argument('--num_gen_feature', type=int, default=100)

    parser.add_argument('--num_dis_layers', type=int, default=2)
    parser.add_argument('--num_dis_hidden', type=int, default=200)

    parser.add_argument('--noise', type=float, default=0.2)

    parser.add_argument('--optimizer', type=str, default='AdamOptimizer',
                        choices=['GradientDescentOptimizer', 'AdamOptimizer', 'AdadeltaOptimizer'])
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--l2norm', type=float, default=0.00001)

    args = parser.parse_args()
    opt.use_argument(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args


if __name__ == '__main__':
    args = get_args()

    assert args.data
    opt.DATA_INFO = json.loads(str(np.load(args.data)['info']))

    if args.sample > 0:
        sample(args.sample, Model(), args.load, output_filename=args.output)

    else:
        logger.auto_set_dir(name=args.exp_name)
        GANTrainer(
            input=QueueInput(get_data(args.data)),
            model=Model()
        ).train_with_defaults(
            callbacks=[ModelSaver(), ],
            steps_per_epoch=args.steps_per_epoch,
            max_epoch=args.max_epoch,
            session_init=SaverRestore(args.load) if args.load else None
        )
