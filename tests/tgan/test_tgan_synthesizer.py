from unittest import TestCase, skip
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal
from tensorflow.test import TestCase as TensorFlowTestCase
from tensorpack.tfutils.tower import TowerContext

from tgan.tgan_synthesizer import Model, get_data, sample


def configure_opt(opt_mock):
    """Set the required opt attributes."""
    opt_mock.batch_size = 50
    opt_mock.z_dim = 50
    opt_mock.sample = 1
    opt_mock.noise = 0.1
    opt_mock.l2norm = 0.001
    opt_mock.num_gen_rnn = 100
    opt_mock.num_gen_feature = 100
    opt_mock.num_dis_layers = 1
    opt_mock.num_dis_hidden = 100

    return opt_mock


class TestModel(TensorFlowTestCase):

    @staticmethod
    def check_operation_nodes(graph, name, node_type, dtype, shape, consumers):
        """Test a graph node parameters.

        Args:
            graph(tf): Graph object the node belongs to.
            name(str): Name of the node.
            node_type(str): Operation type of the node.
            dtype(tf.Dtype): Dtype of the output tensor.
            shape(tuple[int]): Shape of the output tensor.
            consumers(list[str]): List of names of nodes consuming the node's output.

        Returns:
            None.

        Raises:
            AssertionError: If any check fail.

        """
        operation = graph.get_operation_by_name(name)
        assert len(operation.outputs) == 1
        output = operation.outputs[0]

        assert operation.type == node_type
        assert output.dtype == dtype
        assert output.shape.as_list() == shape
        assert output.consumers() == [graph.get_operation_by_name(cons) for cons in consumers]

    @patch('tgan.tgan_synthesizer.InputDesc', autospec=True)
    @patch('tgan.tgan_synthesizer.opt', autospec=True)
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

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test__get_inputs_raises(self, opt_mock):
        """_get_inputs raises a ValueError if an invalid column type is found."""
        # Setup
        instance = MagicMock()
        opt_mock.DATA_INFO = {
            'details': [{'type': 'some invalid type'}]
        }

        expected_message = (
            "opt.DATA_INFO['details'][0]['type'] must be either `category` or `values`. "
            "Instead it was some invalid type."
        )

        try:
            # Run
            Model._get_inputs(instance)

        except ValueError as error:
            # Check
            assert len(error.args) == 1
            assert error.args[0] == expected_message

    def test_compute_kl(self):
        """ """
        # Setup
        real = np.array([1.0, 1.0])
        pred = np.array([0.0, 1.0])

        expected_result = np.array([0.0, 0.0])

        # Run
        with self.test_session():
            with TowerContext('', is_training=False):
                result = Model.compute_kl(real, pred).eval()

        # Check
        assert_equal(result, expected_result)

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_generator_category_column(self, opt_mock):
        """build the graph for the generator model with a single categorical column."""
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = configure_opt(opt_mock)
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'category',
                    'n': 5
                }
            ]
        }

        # Run
        result = instance.generator(z)

        # Check
        assert len(result) == 1
        tensor = result[0]

        assert tensor.name == 'LSTM/00/FC2/output:0'
        assert tensor.dtype == tf.float32
        assert tensor.shape.as_list() == [50, 5]

        assert opt_mock.call_args_list == []
        assert opt_mock.method_calls == []

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_generator_value_column(self, opt_mock):
        """build the graph for the generator model with a single value column."""
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = configure_opt(opt_mock)
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                }
            ]
        }

        # Run
        result = instance.generator(z)

        # Check
        assert len(result) == 2
        first_tensor, second_tensor = result

        assert first_tensor.name == 'LSTM/00/FC2/output:0'
        assert first_tensor.dtype == tf.float32
        assert first_tensor.shape.as_list() == [50, 1]

        assert second_tensor.name == 'LSTM/01/FC2/output:0'
        assert second_tensor.dtype == tf.float32
        assert second_tensor.shape.as_list() == [50, 5]

        assert opt_mock.call_args_list == []
        assert opt_mock.method_calls == []

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_generator_raises(self, opt_mock):
        """If the metadata is has invalid values, an exception is raised."""
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = configure_opt(opt_mock)
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'some invalid type',
                    'n': 5
                }
            ]
        }
        expected_message = (
            "opt.DATA_INFO['details'][0]['type'] must be either `category` or `values`. "
            "Instead it was some invalid type."
        )

        try:  # Run
            instance.generator(z)

        except ValueError as error:  # Check
            assert len(error.args) == 1
            assert error.args[0] == expected_message

    def test_batch_diversity(self):
        """ """
        # Setup
        layer = tf.Variable(np.zeros(15))
        n_kernel = 20
        kernel_dim = 30

        expected_result = np.full((15, 20), 15.0)

        # Run
        result = Model.batch_diversity(layer, n_kernel, kernel_dim)

        # Check - Output properties
        assert result.name == 'Sum_1:0'
        assert result.dtype == tf.float64
        assert result.shape.as_list() == [15, 20]

        graph = result.graph

        # Check - Nodes
        self.check_operation_nodes(
            graph, 'fc_diversity/output', 'Identity', tf.float64, [15, 600], ['Reshape'])
        self.check_operation_nodes(
            graph, 'Reshape', 'Reshape', tf.float64, [15, 20, 30], ['Reshape_1', 'Reshape_2'])
        self.check_operation_nodes(
            graph, 'Reshape_1', 'Reshape', tf.float64, [15, 1, 20, 30], ['sub'])
        self.check_operation_nodes(
            graph, 'Reshape_2', 'Reshape', tf.float64, [1, 15, 20, 30], ['sub'])
        self.check_operation_nodes(graph, 'sub', 'Sub', tf.float64, [15, 15, 20, 30], ['Abs'])
        self.check_operation_nodes(graph, 'Abs', 'Abs', tf.float64, [15, 15, 20, 30], ['Sum'])
        self.check_operation_nodes(graph, 'Sum', 'Sum', tf.float64, [15, 15, 20], ['Neg'])
        self.check_operation_nodes(graph, 'Neg', 'Neg', tf.float64, [15, 15, 20], ['Exp'])
        self.check_operation_nodes(graph, 'Exp', 'Exp', tf.float64, [15, 15, 20], ['Sum_1'])
        self.check_operation_nodes(graph, 'Sum_1', 'Sum', tf.float64, [15, 20], [])

        with self.test_session():
            with TowerContext('', is_training=False):
                tf.initialize_all_variables().run()
                result = result.eval()

        assert_equal(result, expected_result)

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_discriminator(self, opt_mock):
        """ """
        # Setup
        opt_mock = configure_opt(opt_mock)
        opt_mock.num_dis_layers = 1
        instance = Model()
        vecs = [
            np.zeros((7, 10)),
            np.ones((7, 10))
        ]

        # Run
        with TowerContext('', is_training=False):
            result = instance.discriminator(vecs)

        # Check
        assert result.name == 'dis_fc_top/output:0'
        assert result.shape.as_list() == [7, 1]
        assert result.dtype == tf.float64

        graph = result.graph

        self.check_operation_nodes(
            graph, 'concat', 'ConcatV2', tf.float64, [7, 20], ['dis_fc0/fc/Reshape'])

        self.check_operation_nodes(
            graph, 'dis_fc0/fc/output', 'Identity', tf.float64, [7, 100],
            ['dis_fc0/fc_diversity/Reshape', 'dis_fc0/concat']
        )
        self.check_operation_nodes(
            graph, 'dis_fc0/fc_diversity/output', 'Identity', tf.float64, [7, 100],
            ['dis_fc0/Reshape']
        )
        self.check_operation_nodes(
            graph, 'dis_fc0/concat', 'ConcatV2', tf.float64, [7, 110],
            ['dis_fc0/bn/batchnorm/mul']
        )
        self.check_operation_nodes(
            graph, 'dis_fc0/dropout/Identity', 'Identity', tf.float64, [7, 110],
            ['dis_fc0/LeakyRelu/mul', 'dis_fc0/LeakyRelu']
        )

    @skip
    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test__build_graph(self, opt_mock):
        """ """
        # Setup
        opt_mock = configure_opt(opt_mock)
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                }
            ]
        }
        instance = Model()
        inputs = [np.full((50, 10), 0.0), np.full((50, 5), 1.0)]

        # Run
        with TowerContext('', is_training=False):
            result = instance._build_graph(inputs)

        # Check
        assert result is None

    @skip
    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_build_losses(self, opt_mock):
        """ """
        # Setup
        opt_mock = configure_opt(opt_mock)
        logits_real = np.zeros((10, 10), dtype=np.float32)
        logits_fake = np.zeros((10, 10), dtype=np.float32)
        extra_g = 1
        l2_norm = 0.001
        instance = Model()

        # Run
        result = instance.build_losses(logits_real, logits_fake, extra_g, l2_norm)

        # Check
        assert result is None


class TestGetData(TestCase):

    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    @patch('tgan.tgan_synthesizer.BatchData', autospec=True)
    @patch('tgan.tgan_synthesizer.NpDataFlow', autospec=True)
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


class TestSample(TestCase):

    @patch('tgan.tgan_synthesizer.np.savez', autospec=True)
    @patch('tgan.tgan_synthesizer.json.dumps', autospec=True)
    @patch('tgan.tgan_synthesizer.np.concatenate', autospec=True)
    @patch('tgan.tgan_synthesizer.SimpleDatasetPredictor', autospec=True)
    @patch('tgan.tgan_synthesizer.RandomZData', autospec=True)
    @patch('tgan.tgan_synthesizer.PredictConfig', autospec=True)
    @patch('tgan.tgan_synthesizer.get_model_loader', autospec=True)
    @patch('tgan.tgan_synthesizer.opt', autospec=True)
    def test_sample_value_column(
        self, opt_mock, get_model_mock, predict_mock, random_mock,
        simple_mock, concat_mock, json_mock, save_mock
    ):
        """ """
        # Setup
        n = 50
        model = 'Model instance'
        model_path = 'model path'
        output_name = 'output name'
        output_filename = 'output filename'

        opt_mock = configure_opt(opt_mock)
        opt_mock.DATA_INFO = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                },
                {
                    'type': 'category',
                    'n': 5
                }
            ]
        }
        get_model_mock.return_value = 'restored model'
        predict_mock.return_value = 'predict config object'
        simple_instance = MagicMock(**{'get_result.return_value': [[0], [1]]})
        simple_mock.return_value = simple_instance
        random_mock.return_value = 'random z data'
        json_mock.return_value = 'metadata'
        concat_mock.side_effect = [np.zeros((5, 10)), 'concatenated results']

        expected_concat_first_call_args = (([0],), {'axis': 0})

        # Run
        result = sample(n, model, model_path, output_name, output_filename)

        # Check
        assert result is None

        assert opt_mock.call_args_list == []
        get_model_mock.assert_called_once_with('model path')
        predict_mock.assert_called_once_with(
            session_init='restored model',
            model='Model instance',
            input_names=['z'],
            output_names=['output name', 'z']
        )
        random_mock.assert_called_once_with((50, 50))
        simple_mock.assert_called_once_with('predict config object', 'random z data')

        assert len(concat_mock.call_args_list) == 2
        first_call, second_call = concat_mock.call_args_list
        assert first_call == expected_concat_first_call_args
        assert len(second_call[0]) == 1
        assert_equal(second_call[0][0][0], np.zeros((5, 1)))
        assert_equal(second_call[0][0][1], np.zeros((5, 5)))
        second_call[1] == {'axis': 1}

        assert len(save_mock.call_args_list) == 1
        call_args = save_mock.call_args_list[0]
        assert call_args[0] == (output_filename, )
        assert call_args[1]['info'] == 'metadata'
        assert call_args[1]['f00'] == 'concatenated results'
        assert_equal(call_args[1]['f01'], np.zeros((5, 1)))
