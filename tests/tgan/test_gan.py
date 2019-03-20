from unittest import expectedFailure
from unittest.mock import MagicMock, patch

import tensorflow as tf
from tensorflow.test import TestCase as TensorFlowTestCase
from tensorpack.tfutils.tower import TowerFuncWrapper

from tgan.gan import GANModelDesc, GANTrainer, MultiGPUGANTrainer, SeparateGANTrainer


class TestGanModelDesc(TensorFlowTestCase):

    @patch('tgan.gan.tf.get_collection', autospec=True)
    def test_collect_variables(self, collection_mock):
        """collect_variable assign the collected variables defined in the given scopes."""
        # Setup
        g_scope = 'first_scope'
        d_scope = 'second_scope'

        instance = GANModelDesc()

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

        instance = GANModelDesc()

        expected_error_message = 'There are no variables defined in some of the given scopes'

        # Run
        try:
            instance.collect_variables(g_scope, d_scope)

        # Check
        except ValueError as error:
            assert len(error.args) == 1
            assert error.args[0] == expected_error_message

    @expectedFailure
    def test_build_losses(self):
        """ """
        # Setup
        instance = GANModelDesc()
        logits_real = 0.1
        logits_fake = 0.01
        extra_g = 0.2
        l2_norm = 0.00001

        # Run
        result = instance.build_losses(logits_real, logits_fake, extra_g, l2_norm)

        # Check
        assert result


class TestGanTrainer(TensorFlowTestCase):

    @patch('tgan.gan.tf.control_dependencies', autospec=True)
    @patch('tgan.gan.tf.clip_by_value', autospec=True)
    @patch('tgan.gan.TowerContext', autospec=True)
    @patch('tgan.gan.GANTrainer.register_callback', autospec=True)
    @patch('tgan.gan.TowerFuncWrapper', autospec=True)
    def test___init__(self, funcwrapper_mock, register_mock, ctx_mock, clip_mock, control_mock):
        """On init, the model is check, callbacks registered and training iteration defined."""
        # Setup
        input_values = MagicMock(**{
            'setup.return_value': 'setup_value',
            'get_input_tensors.return_value': ['input', 'tensors']
        })
        opt_mock = MagicMock(**{
            'compute_gradients.return_value': [('computed', 'gradients')],
            'apply_gradients.return_value': 'applied gradients'
        })
        model = MagicMock(**{
            'build_graph': 'graph callback',
            'g_loss': 'g_loss',
            'd_loss': 'd_loss',
            'g_vars': 'g_vars',
            'd_vars': 'd_vars',
            '__class__': GANModelDesc,
            'get_optimizer.return_value': opt_mock,
            'get_inputs_desc.return_value': 'inputs_desc'
        })

        tower_wrapped = MagicMock(**{
            '__class__': TowerFuncWrapper
        })
        funcwrapper_mock.return_value = tower_wrapped
        clip_mock.return_value = 'clipped values'

        expected_clip_mock_call_arg_list = [
            (('computed', -5.0, 5.0), {}),
            (('computed', -5.0, 5.0), {}),
        ]

        expected_op_compute_call_args_list = [
            (('g_loss',), {'var_list': 'g_vars'}),
            (('d_loss',), {'var_list': 'd_vars'}),
        ]

        expected_op_apply_call_args_list = [
            (([('clipped values', 'gradients')],), {'name': 'g_op'}),
            (([('clipped values', 'gradients')],), {'name': 'd_op'})
        ]

        # Run
        instance = GANTrainer(input_values, model)

        # Check
        assert instance.model == model
        assert instance.tower_func == tower_wrapped
        assert instance.train_op == 'applied gradients'

        assert clip_mock.call_args_list == expected_clip_mock_call_arg_list
        assert opt_mock.compute_gradients.call_args_list == expected_op_compute_call_args_list
        assert opt_mock.apply_gradients.call_args_list == expected_op_apply_call_args_list

        input_values.setup.assert_called_once_with('inputs_desc')
        input_values.get_input_tensors.assert_called_once_with()

        model.get_inputs_desc.assert_called_once_with()
        model.get_optimizer.assert_called_once_with()

        funcwrapper_mock.assert_called_once_with('graph callback', 'inputs_desc')
        register_mock.assert_called_once_with(instance, 'setup_value')
        ctx_mock.assert_called_once_with('', is_training=True)
        control_mock.assert_called_once_with(['applied gradients'])

    def test___init__raises_value_error(self):
        """If model argument is not a GanModelDesc instance, an error is raised."""
        # Setup
        input_values = 'input'
        model = 'model'

        try:
            # Run
            GANTrainer(input_values, model)

        except ValueError as error:
            # Check
            assert len(error.args) == 1
            error.args[0] == 'Model argument is expected to be an instance of GanModelDesc.'


class TestSeparateGanTrainer(TensorFlowTestCase):

    @patch('tgan.gan.TowerContext', autospec=True)
    @patch('tgan.gan.SeparateGANTrainer.register_callback', autospec=True)
    @patch('tgan.gan.TowerFuncWrapper', autospec=True)
    def test___init__(self, funcwrapper_mock, register_mock, ctx_mock):
        """On init, callbacks are set and the training iteration defined."""
        # Setup

        input_values = MagicMock(**{
            'setup.return_value': 'setup_value',
            'get_input_tensors.return_value': ['input', 'tensors']
        })
        opt_mock = MagicMock(**{
            'minimize.side_effect': ['d_min', 'g_min'],
        })
        model = MagicMock(**{
            'build_graph': 'graph callback',
            'g_loss': 'g_loss',
            'd_loss': 'd_loss',
            'g_vars': 'g_vars',
            'd_vars': 'd_vars',
            '__class__': GANModelDesc,
            'get_optimizer.return_value': opt_mock,
            'get_inputs_desc.return_value': 'inputs_desc'
        })
        d_period = 1
        g_period = 1

        tower_wrapped = MagicMock(**{
            '__class__': TowerFuncWrapper
        })
        funcwrapper_mock.return_value = tower_wrapped

        # Run
        instance = SeparateGANTrainer(input_values, model, d_period, g_period)

        # Check
        assert instance._d_period == 1
        assert instance._g_period == 1
        assert instance.d_min == 'd_min'
        assert instance.g_min == 'g_min'

        input_values.setup.assert_called_once_with('inputs_desc')
        input_values.get_input_tensors.assert_called_once_with()

        assert model.get_inputs_desc.call_args_list == [((), {}), ((), {})]
        model.get_optimizer.assert_called_once_with()

        assert opt_mock.minimize.call_args_list == [
            (('d_loss',), {'var_list': 'd_vars', 'name': 'd_min'}),
            (('g_loss',), {'var_list': 'g_vars', 'name': 'g_min'}),
        ]

        funcwrapper_mock.assert_called_once_with('graph callback', 'inputs_desc')
        tower_wrapped.assert_called_once_with('input', 'tensors')
        register_mock.assert_called_once_with(instance, 'setup_value')
        ctx_mock.assert_called_once_with('', is_training=True)

    def test___init__raises(self):
        """If d_period or g_period are lower than 1 an exception is raised. Or if none is 1."""
        # Setup
        input_values = 'input'
        model = 'model'
        d_period = 0
        g_period = 0

        try:
            # Run
            SeparateGANTrainer(input_values, model, d_period, g_period)

        except ValueError as error:
            # Check
            assert len(error.args)
            assert error.args[0] == 'The minimum between d_period and g_period must be 1.'

    def test_run_step(self):
        """Run step used modulo x_period to decide wich steps run."""
        # Setup
        instance = MagicMock(
            spec=SeparateGANTrainer,
            global_step=4,
            _d_period=1,
            _g_period=2,
            d_min='d_min',
            g_min='g_min'
        )

        expected_run_call_arg_list = [
            (('d_min',), {}),
            (('g_min',), {}),
        ]

        # Run
        result = SeparateGANTrainer.run_step(instance)

        # Check
        assert result is None
        assert instance.hooked_sess.run.call_args_list == expected_run_call_arg_list


class TestMultiGpuGanTrainer(TensorFlowTestCase):

    def test___init__(self):
        """ """
        # Setup

        # Run

        # Check

    def test___init__raises(self):
        """If nr_gpu if not greater than 1 and exception is raised."""
        # Setup
        nr_gpu = 0
        input_source = 'input_source'
        model = 'model'

        try:
            # Run
            MultiGPUGANTrainer(nr_gpu, input_source, model)
        except ValueError as error:
            # Check
            assert len(error.args) == 1
            assert error.args[0] == 'nr_gpu must be strictly greater than 1.'
