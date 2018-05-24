#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import glob
import numpy as np
import os
import argparse
import json
# from gumbel import gumbel_softmax
from datetime import datetime


from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf
from np_data_flow import NpDataFlow

from GAN import GANTrainer, RandomZData, GANModelDesc

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
    def _get_inputs(self):
        inputs = []
        for col_id, col_info in enumerate(opt.DATA_INFO['details']):
            if col_info['type'] == 'value':
                gaussian_components = col_info['n']
                inputs.append(InputDesc(tf.float32, (opt.batch_size, 1), 'input%02dvalue' % col_id))
                inputs.append(InputDesc(tf.float32, (opt.batch_size, gaussian_components), 'input%02dcluster' % col_id))
            elif col_info['type'] == 'category':
                inputs.append(InputDesc(tf.int32, (opt.batch_size, 1), 'input%02d' % col_id))
            else:
                assert 0
        return inputs

    def generator(self, z):
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(opt.num_gen_rnn)

            state = cell.zero_state(opt.batch_size, dtype='float32')
            attention = tf.zeros(shape=(opt.batch_size, opt.num_gen_rnn), dtype='float32')
            input = tf.get_variable(name='go', shape=(1, opt.num_gen_feature)) # <GO>
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
        def batch_diversity(l, n_kernel=10, kernel_dim=10):
            M = FullyConnected('fc_diversity', l, n_kernel * kernel_dim, nl=tf.identity)
            M = tf.reshape(M, [-1, n_kernel, kernel_dim])
            M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
            M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
            diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
            diversity = tf.reduce_sum(diff, axis=0)
            return diversity


        """ return a (b, 1) logits"""
        l = tf.concat(vecs, axis=1)
        for i in range(opt.num_dis_layers):
            with tf.variable_scope('dis_fc{}'.format(i)):
                if i == 0:
                    l = FullyConnected('fc', l, opt.num_dis_hidden, nl=tf.identity,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                else:
                    l = FullyConnected('fc', l, opt.num_dis_hidden, nl=tf.identity)
                l = tf.concat([l, batch_diversity(l)], axis=1)
                l = BatchNorm('bn', l, center=True, scale=False)
                l = Dropout(l)
                l = tf.nn.leaky_relu(l)

        l = FullyConnected('dis_fc_top', l, 1, nl=tf.identity)
        return l

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
            vecs_output = tf.identity(
                tf.concat(vecs_denorm, axis=1), name='gen')

        vecs_pos = []
        ptr = 0
        for col_id, col_info in enumerate(opt.DATA_INFO['details']):
            if col_info['type'] == 'category':
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])
                noise_input = one_hot
                if opt.sample == 0:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=opt.noise)
                    noise_input = (one_hot + noise) / tf.reduce_sum(one_hot + noise, keep_dims=True, axis=1)
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
    ds = NpDataFlow(datafile, shuffle=True)
    opt.distribution = ds.distribution
    ds = BatchData(ds, opt.batch_size)
    return ds


def sample(n, model, model_path, output_name='gen/gen', output_filename=None):
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
            features['f%02d' % col_id] = results[:, ptr:ptr+1]
            ptr += 1
        elif col_info['type'] == 'value':
            gaussian_components = col_info['n']
            val = results[:, ptr:ptr+1]
            ptr += 1
            pro = results[:, ptr:ptr+gaussian_components]
            ptr += gaussian_components
            features['f%02d' % col_id] = np.concatenate([val, pro], axis=1)
        else:
            assert 0
    np.savez(output_filename, info=json.dumps(opt.DATA_INFO), **features)

def get_args():
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
            model=Model()).train_with_defaults(
            callbacks=[
                ModelSaver(),
            ],
            steps_per_epoch=args.steps_per_epoch,
            max_epoch=args.max_epoch,
            session_init=SaverRestore(args.load) if args.load else None
        )
