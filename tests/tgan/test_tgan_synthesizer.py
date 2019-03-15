from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal
from tensorflow.test import TestCase as TensorFlowTestCase
from tensorpack.tfutils.tower import TowerContext

from tgan.TGAN_synthesizer import Model, get_data


class TestModel(TensorFlowTestCase):

    def configure_opt(self, opt_mock):
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

    @patch('tgan.TGAN_synthesizer.InputDesc', autospec=True)
    @patch('tgan.TGAN_synthesizer.opt', autospec=True)
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

    @patch('tgan.TGAN_synthesizer.opt', autospec=True)
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

    @patch('tgan.TGAN_synthesizer.opt')
    def test_generator_category_column(self, opt_mock):
        """build the graph for the generator model."""
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = self.configure_opt(opt_mock)
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

    @patch('tgan.TGAN_synthesizer.opt')
    def test_generator_value_column(self, opt_mock):
        """build the graph for the generator model."""
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = self.configure_opt(opt_mock)
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

    @patch('tgan.TGAN_synthesizer.opt')
    def test_generator_raises(self, opt_mock):
        """ """
        # Setup
        instance = Model()
        z = np.zeros((50, 100))

        opt_mock = self.configure_opt(opt_mock)
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

        # Check
        assert result.name == 'Sum_1:0'
        assert result.dtype == tf.float64
        assert result.shape.as_list() == [15, 20]

        import ipdb; ipdb.set_trace()
        print('')

        ops = ["""
            <tf.Operation 'Variable/initial_value' type=Const>
            <tf.Operation 'Variable' type=VariableV2>
            <tf.Operation 'Variable/Assign' type=Assign>
            <tf.Operation 'Variable/read' type=Identity>
            <tf.Operation 'fc_diversity/Reshape/shape' type=Const>
            <tf.Operation 'fc_diversity/Reshape' type=Reshape>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal/shape' type=Const>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal/mean' type=Const>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal/stddev' type=Const>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal/TruncatedNormal' type=TruncatedNormal>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal/mul' type=Mul>
            <tf.Operation 'fc_diversity/W/Initializer/truncated_normal' type=Add>
            <tf.Operation 'fc_diversity/W' type=VariableV2>
            <tf.Operation 'fc_diversity/W/Assign' type=Assign>
            <tf.Operation 'fc_diversity/W/read' type=Identity>
            <tf.Operation 'fc_diversity/b/Initializer/zeros' type=Const>
            <tf.Operation 'fc_diversity/b' type=VariableV2>
            <tf.Operation 'fc_diversity/b/Assign' type=Assign>
            <tf.Operation 'fc_diversity/b/read' type=Identity>
            <tf.Operation 'fc_diversity/MatMul' type=MatMul>
            <tf.Operation 'fc_diversity/BiasAdd' type=BiasAdd>
            <tf.Operation 'fc_diversity/Identity' type=Identity>
            <tf.Operation 'fc_diversity/output' type=Identity>
            <tf.Operation 'Reshape/shape' type=Const>
            <tf.Operation 'Reshape' type=Reshape>
            <tf.Operation 'Reshape_1/shape' type=Const>
            <tf.Operation 'Reshape_1' type=Reshape>
            <tf.Operation 'Reshape_2/shape' type=Const>
            <tf.Operation 'Reshape_2' type=Reshape>
            <tf.Operation 'sub' type=Sub>
            <tf.Operation 'Abs' type=Abs>
            <tf.Operation 'Sum/reduction_indices' type=Const>
            <tf.Operation 'Sum' type=Sum>
            <tf.Operation 'Neg' type=Neg>
            <tf.Operation 'Exp' type=Exp>
            <tf.Operation 'Sum_1/reduction_indices' type=Const>
            <tf.Operation 'Sum_1' type=Sum>"""
        ]

        with self.test_session():
            with TowerContext('', is_training=False):
                tf.initialize_all_variables().run()
                result = result.eval()

        assert_equal(result, expected_result)

    @patch('tgan.TGAN_synthesizer.opt')
    def test_discriminator(self, opt_mock):
        """ """
        # Setup
        opt_mock = self.configure_opt(opt_mock)
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
        
        concat_op = graph.get_operation_by_name('concat')
        assert concat_op.type == 'ConcatV2'
        concat_output = concat_op.outputs
        assert len(concat_output) == 1
        assert concat_output[0].name == 'concat:0'
        assert concat_output[0].dtype == tf.float64
        assert concat_output[0].shape.as_list() == [7, 20]

        fully_connected_op = graph.get_operation_by_name('dis_fc0/fc/output')
        assert fully_connected_op.type == 'Identity'
        fully_connected_output = fully_connected_op.outputs
        assert len(fully_connected_output) == 1
        assert fully_connected_output[0].name == 'dis_fc0/fc/output:0'
        assert fully_connected_output[0].dtype == tf.float64
        # The shape came from vecs shape[1] and opt_mock.num_dis_hidden
        assert fully_connected_output[0].shape.as_list() == [7, 100] 

        diversity_op = graph.get_operation_by_name('dis_fc0/fc_diversity/output')
        assert diversity_op.type == 'Identity'
        diversity_output = diversity_op.outputs
        assert len(diversity_output) == 1
        assert diversity_output[0].name == 'dis_fc0/fc_diversity/output:0'
        assert diversity_output[0].dtype == tf.float64
        assert diversity_output[0].shape.as_list() == [7, 100]



        #assert graph.get_operations() == 
        """[

            <tf.Operation 'dis_fc0/Reshape/shape' type=Const>,
            <tf.Operation 'dis_fc0/Reshape' type=Reshape>,
            <tf.Operation 'dis_fc0/Reshape_1/shape' type=Const>,
            <tf.Operation 'dis_fc0/Reshape_1' type=Reshape>,
            <tf.Operation 'dis_fc0/Reshape_2/shape' type=Const>,
            <tf.Operation 'dis_fc0/Reshape_2' type=Reshape>,
            <tf.Operation 'dis_fc0/sub' type=Sub>,
            <tf.Operation 'dis_fc0/Abs' type=Abs>,
            <tf.Operation 'dis_fc0/Sum/reduction_indices' type=Const>,
            <tf.Operation 'dis_fc0/Sum' type=Sum>,
            <tf.Operation 'dis_fc0/Neg' type=Neg>,
            <tf.Operation 'dis_fc0/Exp' type=Exp>,
            <tf.Operation 'dis_fc0/Sum_1/reduction_indices' type=Const>,
            <tf.Operation 'dis_fc0/Sum_1' type=Sum>,
            <tf.Operation 'dis_fc0/concat/axis' type=Const>,
            <tf.Operation 'dis_fc0/concat' type=ConcatV2>,
            <tf.Operation 'dis_fc0/bn/beta/Initializer/zeros' type=Const>,
            <tf.Operation 'dis_fc0/bn/beta' type=VariableV2>,
            <tf.Operation 'dis_fc0/bn/beta/Assign' type=Assign>,
            <tf.Operation 'dis_fc0/bn/beta/read' type=Identity>,
            <tf.Operation 'dis_fc0/bn/mean/EMA/Initializer/zeros' type=Const>,
            <tf.Operation 'dis_fc0/bn/mean/EMA' type=VariableV2>,
            <tf.Operation 'dis_fc0/bn/mean/EMA/Assign' type=Assign>,
            <tf.Operation 'dis_fc0/bn/mean/EMA/read' type=Identity>,
            <tf.Operation 'dis_fc0/bn/variance/EMA/Initializer/ones' type=Const>,
            <tf.Operation 'dis_fc0/bn/variance/EMA' type=VariableV2>,
            <tf.Operation 'dis_fc0/bn/variance/EMA/Assign' type=Assign>,
            <tf.Operation 'dis_fc0/bn/variance/EMA/read' type=Identity>,
            <tf.Operation 'dis_fc0/bn/batchnorm/add/y' type=Const>,
            <tf.Operation 'dis_fc0/bn/batchnorm/add' type=Add>,
            <tf.Operation 'dis_fc0/bn/batchnorm/Rsqrt' type=Rsqrt>,
            <tf.Operation 'dis_fc0/bn/batchnorm/mul' type=Mul>,
            <tf.Operation 'dis_fc0/bn/batchnorm/mul_1' type=Mul>,
            <tf.Operation 'dis_fc0/bn/batchnorm/sub' type=Sub>,
            <tf.Operation 'dis_fc0/bn/batchnorm/add_1' type=Add>,
            <tf.Operation 'dis_fc0/bn/output' type=Identity>,
            <tf.Operation 'dis_fc0/dropout/Identity' type=Identity>,
            <tf.Operation 'dis_fc0/LeakyRelu/alpha' type=Const>,
            <tf.Operation 'dis_fc0/LeakyRelu/mul' type=Mul>,
            <tf.Operation 'dis_fc0/LeakyRelu' type=Maximum>,
            <tf.Operation 'dis_fc_top/Reshape/shape' type=Const>,
            <tf.Operation 'dis_fc_top/Reshape' type=Reshape>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal/shape' type=Const>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal/mean' type=Const>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal/stddev' type=Const>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal/TruncatedNormal' type=TruncatedNormal>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal/mul' type=Mul>,
            <tf.Operation 'dis_fc_top/W/Initializer/truncated_normal' type=Add>,
            <tf.Operation 'dis_fc_top/W' type=VariableV2>,
            <tf.Operation 'dis_fc_top/W/Assign' type=Assign>,
            <tf.Operation 'dis_fc_top/W/read' type=Identity>,
            <tf.Operation 'dis_fc_top/b/Initializer/zeros' type=Const>,
            <tf.Operation 'dis_fc_top/b' type=VariableV2>,
            <tf.Operation 'dis_fc_top/b/Assign' type=Assign>,
            <tf.Operation 'dis_fc_top/b/read' type=Identity>,
            <tf.Operation 'dis_fc_top/MatMul' type=MatMul>,
            <tf.Operation 'dis_fc_top/BiasAdd' type=BiasAdd>,
            <tf.Operation 'dis_fc_top/Identity' type=Identity>,
            <tf.Operation 'dis_fc_top/output' type=Identity>
        ]"""


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
