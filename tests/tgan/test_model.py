from unittest.mock import patch

import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal
from tensorflow.test import TestCase as TensorFlowTestCase
from tensorpack.tfutils.tower import TowerContext

from tgan.model import GraphBuilder, TGANModel


class TestGraphBuilder(TensorFlowTestCase):

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

    @patch('tgan.model.InputDesc', autospec=True)
    def test_inputs(self, input_mock):
        """_get_inputs return a list of all the metadat for input entries in the graph."""
        # Setup
        metadata = {
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
        instance = GraphBuilder(metadata)
        input_mock.side_effect = ['value_input', 'cluster_input', 'category_input']

        expected_input_mock_call_args_list = [
            ((tf.float32, (200, 1), 'input00value'), {}),
            ((tf.float32, (200, 5), 'input00cluster'), {}),
            ((tf.int32, (200, 1), 'input01'), {}),
        ]

        expected_result = ['value_input', 'cluster_input', 'category_input']

        # Run
        result = instance.inputs()

        # Check
        assert result == expected_result
        assert input_mock.call_args_list == expected_input_mock_call_args_list

    def test_inputs_raises(self):
        """_get_inputs raises a ValueError if an invalid column type is found."""
        # Setup
        metadata = {
            'details': [{'type': 'some invalid type'}]
        }

        instance = GraphBuilder(metadata)

        expected_message = (
            "self.metadata['details'][0]['type'] must be either `category` or `values`. "
            "Instead it was some invalid type."
        )

        try:
            # Run
            instance.inputs()

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
                result = GraphBuilder.compute_kl(real, pred).eval()

        # Check
        assert_equal(result, expected_result)

    def test_generator_category_column(self):
        """build the graph for the generator GraphBuilder with a single categorical column."""
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'category',
                    'n': 5
                }
            ]
        }

        instance = GraphBuilder(metadata)
        z = np.zeros((instance.batch_size, 100))

        # Run
        result = instance.generator(z)

        # Check
        assert len(result) == 1
        tensor = result[0]

        assert tensor.name == 'LSTM/00/FC2/output:0'
        assert tensor.dtype == tf.float32
        assert tensor.shape.as_list() == [200, 5]

    def test_generator_value_column(self):
        """build the graph for the generator GraphBuilder with a single value column."""
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                }
            ]
        }

        instance = GraphBuilder(metadata)
        z = np.zeros((instance.batch_size, 100))

        # Run
        result = instance.generator(z)

        # Check
        assert len(result) == 2
        first_tensor, second_tensor = result

        assert first_tensor.name == 'LSTM/00/FC2/output:0'
        assert first_tensor.dtype == tf.float32
        assert first_tensor.shape.as_list() == [200, 1]

        assert second_tensor.name == 'LSTM/01/FC2/output:0'
        assert second_tensor.dtype == tf.float32
        assert second_tensor.shape.as_list() == [200, 5]

    def test_generator_raises(self):
        """If the metadata is has invalid values, an exception is raised."""
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'some invalid type',
                    'n': 5
                }
            ]
        }

        instance = GraphBuilder(metadata)
        z = np.zeros((instance.batch_size, 100))

        expected_message = (
            "self.metadata['details'][0]['type'] must be either `category` or `values`. "
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
        result = GraphBuilder.batch_diversity(layer, n_kernel, kernel_dim)

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

    def test_discriminator(self):
        """ """
        # Setup
        metadata = {}
        instance = GraphBuilder(metadata, num_dis_layers=1)
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

    def test_build_graph(self):
        """ """
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'value',
                    'n': 5
                }
            ]
        }
        instance = GraphBuilder(metadata)
        inputs = [
            np.full((50, 5), 0.0, dtype=np.float32),
            np.full((50, 1), 1.0, dtype=np.float32)
        ]

        # Run
        with TowerContext('', is_training=False):
            result = instance.build_graph(*inputs)

        # Check
        assert result is None

    def test_build_losses(self):
        """build_losses generates the loss function for both components."""
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'value',
                    'n': 5,
                },
                {
                    'type': 'category',
                    'n': 2
                }
            ],
            'num_columns': 2
        }
        instance = GraphBuilder(metadata)
        logits_real = np.zeros((10, 10), dtype=np.float32)
        logits_fake = np.zeros((10, 10), dtype=np.float32)
        extra_g = 1.0
        l2_norm = 0.001

        inputs = [
            np.full((200, 1), 0.0, dtype=np.float32),
            np.full((200, 5), 1.0, dtype=np.float32),
            np.full((200, 1), 0)
        ]
        with TowerContext('', is_training=False):
            instance.build_graph(*inputs)

            # Run
            result = instance.build_losses(logits_real, logits_fake, extra_g, l2_norm)

        # Check
        assert result is None

    @patch('tgan.model.tf.get_collection', autospec=True)
    def test_collect_variables(self, collection_mock):
        """collect_variable assign the collected variables defined in the given scopes."""
        # Setup
        g_scope = 'first_scope'
        d_scope = 'second_scope'

        opt = None
        instance = GraphBuilder(opt)

        collection_mock.side_effect = [['variables for g_scope'], ['variables for d_scope']]

        expected_g_vars = ['variables for g_scope']
        expected_d_vars = ['variables for d_scope']
        expected_collection_mock_call_args_list = [
            ((tf.GraphKeys.TRAINABLE_VARIABLES, 'first_scope'), {}),
            ((tf.GraphKeys.TRAINABLE_VARIABLES, 'second_scope'), {})
        ]

        # Run
        instance.collect_variables(g_scope, d_scope)

        # Check
        assert instance.g_vars == expected_g_vars
        assert instance.d_vars == expected_d_vars

        assert collection_mock.call_args_list == expected_collection_mock_call_args_list

    def test_collect_variables_raises_value_error(self):
        """If no variables are found on one scope, a ValueError is raised."""
        # Setup
        g_scope = 'first_scope'
        d_scope = 'second_scope'

        metadata = None
        instance = GraphBuilder(metadata)

        expected_error_message = 'There are no variables defined in some of the given scopes'

        # Run
        try:
            instance.collect_variables(g_scope, d_scope)

        # Check
        except ValueError as error:
            assert len(error.args) == 1
            assert error.args[0] == expected_error_message


class TestTGANModel(TensorFlowTestCase):

    def test___init__(self):
        """On init, arguments are set as attributes."""
        # Setup
        continuous_columns = []

        # Run
        instance = TGANModel(continuous_columns)

        # Check
        assert instance.continuous_columns == continuous_columns
        assert instance.log_dir == 'output/logs'
        assert instance.model_dir == 'output/model'
        assert instance.max_epoch == 5
        assert instance.steps_per_epoch == 10000
        assert instance.batch_size == 200
        assert instance.z_dim == 200
        assert instance.gpu is None
        assert instance.save_checkpoints is True
        assert instance.restore_session is True
