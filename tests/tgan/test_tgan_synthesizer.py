from unittest import TestCase, expectedFailure
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf
from tensorflow.test import TestCase as TensorFlowTestCase
from tensorpack.tfutils.tower import TowerContext
from tensorpack.utils.globvars import globalns as opt

from tgan.TGAN_synthesizer import Model, get_data


class TestModel(TensorFlowTestCase):

    @patch('tgan.TGAN_synthesizer.InputDesc')
    @patch('tgan.TGAN_synthesizer.opt')
    def test__get_inputs(self, opt_mock, input_mock):
        """_get_inputs return a list of all the metadat for input entries in the graph."""
        # Setup
        instance = MagicMock()

        opt_mock.batch_size = 10
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'value',
                    'n': 5,
                },
                {
                    'type': 'category'
                }
            ]
        }
        input_mock.side_effect = ['value_input', 'cluster_input', 'category_input']

        expected_input_mock_call_args_list = [
            ((tf.float32, (10, 1), 'input00value'), {}),
            ((tf.float32, (10, 5), 'input00cluster'), {}),
            ((tf.int32, (10, 1), 'input01'), {}),
        ]

        expected_result = ['value_input', 'cluster_input', 'category_input']

        # Run
        result = Model._get_inputs(instance)

        # Check
        assert result == expected_result

        assert instance.call_args_list == []
        assert instance.method_calls == []
        assert opt_mock.call_args_list == []
        assert input_mock.call_args_list == expected_input_mock_call_args_list

    @expectedFailure
    def test_build_losses(self):
        """ """
        # Setup
        instance = Model()
        logits_real = 0.1
        logits_fake = 0.01
        extra_g = 0.2
        l2_norm = 0.00001

        # Run
        result = instance.build_losses(logits_real, logits_fake, extra_g, l2_norm)

        # Check
        assert result

    @expectedFailure
    def test_get_optimizer(self):
        """Return a decorated version of _get_optimizer with unlimited cache."""
        # Setup
        instance = Model()

        # Run
        result = instance.get_optimizer()

        # Check
        assert result

        opt.DATA_INFO = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                },
                {
                    'type': 'category',
                    'mapping': ['A', 'B', 'C', 'D'],
                    'n': 4
                }
            ],
        }
        opt.batch_size = 50
        opt.z_dim = 50
        opt.num_gen_rnn = 100
        opt.sample = 1
        opt.noise = 0.1
        opt.l2norm = 0.001
        opt.num_gen_feature = 100
        opt.num_dis_layers = 1
        opt.num_dis_hidden = 100
        instance.build_graph(
            np.full((50, 1), 0),
            np.full((50, 5), 0),
            np.full((50, 1), 0)
        )
        with self.test_session():
            with TowerContext('', is_training=False):
                with tf.variable_scope('first_scope'):
                    tf.get_variable('a', 1, trainable=True)

                with tf.variable_scope('second_scope'):
                    tf.get_variable('b', 0, trainable=True)

                instance.collect_variables('first_scope', 'second_scope')


class TestGetData(TestCase):

    @patch('tgan.TGAN_synthesizer.opt', autospec=True)
    @patch('tgan.TGAN_synthesizer.BatchData', autospec=True)
    @patch('tgan.TGAN_synthesizer.NpDataFlow', autospec=True)
    def test_get_data(self, np_mock, batch_mock, opt_mock):
        """get_data returns a Batchdata, but set global opt with NpdataFlow.distribution."""
        # Setup
        datafile = 'path to data'

        data_mock = MagicMock(distribution='distribution')
        np_mock.return_value = data_mock
        batch_mock.return_value = 'batch data'
        opt_mock.batch_size = 10

        # Run
        result = get_data(datafile)

        # Check
        assert result == 'batch data'

        np_mock.assert_called_once_with('path to data', shuffle=True)
        batch_mock.assert_called_once_with(data_mock, 10)
        assert opt_mock.distribution == 'distribution'
        assert data_mock.call_args_list == []
        assert opt_mock.call_args_list == []
