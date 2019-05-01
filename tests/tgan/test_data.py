from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from tgan.data import (
    MultiModalNumberTransformer, RandomZData, TGANDataFlow, check_inputs, check_metadata)


class TestTGANDataFlow(TestCase):

    def test___init__(self):
        """ """
        # Setup
        shuffle = False
        data = {
            'f00': np.zeros((1, 5)),
            'f01': np.ones((1, 5)),
            'info': 'Metadata'
        }
        metadata = {
            'details': [
                {
                    'type': 'category',
                },
                {
                    'type': 'value',
                }
            ],
            'num_features': 1,
        }

        expected_data = [(
            np.array([0, 0, 0, 0, 0], dtype=np.int32),
            np.array([1.]),
            np.array([1., 1., 1., 1.])
        )]

        # Run
        result = TGANDataFlow(data, metadata, shuffle)

        # Check
        assert result.shuffle is False
        assert result.metadata == metadata
        assert result.num_features == 1
        [assert_equal(actual, expected) for actual, expected in zip(result.data, expected_data)]

    def test_size(self):
        """size return the size of self.data."""
        # Setup
        shuffle = False
        metadata = {
            'details': [],
            'num_features': 1,
        }
        data = np.array([1, 2, 3])
        instance = TGANDataFlow([], metadata, shuffle)
        instance.data = data
        # Run
        result = instance.size()

        # Check
        assert result == 3

    def test_get_data(self):
        """get_data return a generator yielding the contents of self.data."""
        # Setup
        shuffle = False
        metadata = {
            'details': [
                {
                    'type': 'category'
                },
                {
                    'type': 'value'
                }
            ],
            'num_features': 2,
        }
        data = {
            'f00': [0, 1, 2],
            'f01': np.array([
                [0.0, 1.0, 2.0, 3.0],
                [0.1, 1.1, 2.1, 3.1],
                [0.2, 1.2, 2.2, 3.2],
            ]),
        }
        instance = TGANDataFlow(data, metadata, shuffle)
        expected_result = [
            (0, np.array([0.0]), np.array([1.0, 2.0, 3.0])),
            (1, np.array([0.1]), np.array([1.1, 2.1, 3.1])),
            (2, np.array([0.2]), np.array([1.2, 2.2, 3.2]))]

        # Run
        generator = instance.get_data()
        result = [item for item in generator]

        # Check
        assert_equal(result, expected_result)

    def test_get_data_shuffle(self):
        """get_data shuffles data when self.suffle is True."""
        # Setup
        shuffle = True
        metadata = {
            'details': [
                {
                    'type': 'category'
                }
            ],
            'num_features': 1,
        }
        data = {'f00': [0, 1, 2, 3, 4]}
        instance = TGANDataFlow(data, metadata, shuffle)

        instance.rng = np.random.RandomState(0)

        # Run
        generator = instance.get_data()
        result = [item for item in generator]

        # Check
        assert result == [(2,), (0,), (1,), (3,), (4,)]


class TestRandomZData(TestCase):

    @patch('tgan.data.DataFlow.__init__', autospec=True)
    def test___init__(self, dataflow_mock):
        """On init, shape is set as attribute and super is called."""
        # Setup
        shape = (10, 2)

        # Run
        instance = RandomZData(shape)

        # Check
        assert instance.shape == (10, 2)
        dataflow_mock.assert_called_once_with()

    @patch('tgan.data.np.random.normal', autospec=True)
    def test_get_data(self, normal_mock):
        """get_data return an infinite generator of normal vectors of the given shape."""
        # Setup
        shape = (2, 1)
        instance = RandomZData(shape)
        normal_mock.return_value = [[0.5], [0.5]]

        # Run
        generator = instance.get_data()
        result = next(generator)

        # Check
        assert result == [normal_mock.return_value]
        normal_mock.assert_called_once_with(0, 1, size=(2, 1))


class TestCheckMetadata(TestCase):

    def test_check_metadata_valid_input(self):
        """If metadata is valid, the function does nothing."""
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'category'
                },
                {
                    'type': 'value'
                }
            ]
        }

        # Run / Check
        message = 'check_metadata does nothing if the metadatada is valid'
        assert check_metadata(metadata) is None, message

    def test_check_metadata_raises_invalid_input(self):
        """ """
        # Setup
        metadata = {
            'details': [
                {
                    'type': 'category'
                },
                {
                    'type': 'value'
                }
            ]
        }

        try:
            # Run
            check_metadata(metadata)
        except AssertionError as error:
            # Check
            assert error.args[0] == 'The given metadata contains unsupported types.'


class TestCheckInputs(TestCase):

    def test_check_inputs_valid_input(self):
        """When the input is a valid np.ndarray, the function is called as is."""
        # Setup
        function = MagicMock(return_value='result')
        instance = MagicMock()
        data = np.zeros((5, 1))

        # Run
        decorated = check_inputs(function)
        result = decorated(instance, data)

        # Check
        assert result == 'result'
        function.assert_called_once_with(instance, data)
        assert instance.call_args_list == []
        assert instance.method_calls == []

    def test_check_inputs_invalid_type(self):
        """If the input of the decorated function is not a np.ndarray an exception is raised."""
        # Setup
        function = MagicMock(return_value='result')
        instance = MagicMock()
        data = 'some invalid data'
        expected_message = 'The argument `data` must be a numpy.ndarray with shape (n, 1).'

        # Run
        decorated = check_inputs(function)
        try:
            decorated(instance, data)

        except ValueError as error:
            # Check
            assert error.args[0] == expected_message

        assert function.call_args_list == []
        assert instance.call_args_list == []
        assert instance.method_calls == []

    def test_check_inputs_invalid_shape(self):
        """If the input of the decorated function has an invalid shape an exception is raised."""
        # Setup
        function = MagicMock(return_value='result')
        instance = MagicMock()
        data = np.array(['some array', 'of invalid shape'])
        expected_message = 'The argument `data` must be a numpy.ndarray with shape (n, 1).'

        # Run
        decorated = check_inputs(function)
        try:
            decorated(instance, data)

        except ValueError as error:
            # Check
            assert error.args[0] == expected_message

        assert function.call_args_list == []
        assert instance.call_args_list == []
        assert instance.method_calls == []


class TestMultiModelNumberTransformer(TestCase):

    def test___init__(self):
        """On init, constructor arguments are set as attributes."""
        # Setup / Run
        num_modes = 10
        instance = MultiModalNumberTransformer(num_modes=num_modes)

        # Check
        assert instance.num_modes == 10

    @patch('tgan.data.GaussianMixture', autospec=True)
    def test_transform(self, gaussian_mock):
        """transform cluster the values using a Gaussian Mixture model."""
        # Setup
        data = np.array([
            [0.1],
            [0.5],
            [1.0]
        ])
        num_modes = 2

        instance = MultiModalNumberTransformer(num_modes)

        model_mock_spec = {
            'fit.return_value': None,
            'means_': np.array([[0.0], [1.0]]),
            'covariances_': np.array([[[4.0], [1.0]]]),
            'predict_proba.return_value': np.array([
                [0.1, 0.2],
                [0.2, 0.1],
                [0.1, 0.2]
            ])
        }

        model_mock = MagicMock(**model_mock_spec)
        gaussian_mock.return_value = model_mock

        expected_features = np.array([
            [-0.45],
            [0.125],
            [0.000]
        ])
        expected_probabilities = np.array([
            [0.1, 0.2],
            [0.2, 0.1],
            [0.1, 0.2]
        ])
        expected_means = np.array([0.0, 1.0])  # Reshape from model_mock.means_
        expected_stds = np.array([2.0, 1.0])   # Reshape and sqrt from model_mock.covariances_

        # Run
        result = instance.transform(data)
        features, probabilities, means, stds = result

        # Check
        assert_equal(features, expected_features)
        assert_equal(probabilities, expected_probabilities)
        assert_equal(means, expected_means)
        assert_equal(stds, expected_stds)

        gaussian_mock.assert_called_once_with(2)
        model_mock.fit.assert_called_once_with(data)
        model_mock.predict_proba.assert_called_once_with(data)

    def test_inverse_transform(self):
        """reverse_transform reverses the clustering."""
        # Setup
        data = np.array([
            [-0.45, 0.1, 0.2],
            [0.125, 0.2, 0.1],
            [0.000, 0.1, 0.2]

        ])

        info = {
            'type': 'value',
            'means': np.array([0.0, 1.0]),
            'stds': np.array([2.0, 1.0]),
        }

        instance = MultiModalNumberTransformer()

        expected_result = np.array([0.1, 0.5, 1.0])

        # Run
        result = instance.inverse_transform(data, info)

        # Check
        assert_allclose(result, expected_result)
