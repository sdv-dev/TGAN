from unittest import TestCase
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_equal

from tgan.np_data_flow import NpDataFlow


class TestNPDataFlow(TestCase):

    @patch('tgan.np_data_flow.json.loads', autospec=True)
    @patch('tgan.np_data_flow.np.load', autospec=True)
    def test___init__(self, load_mock, json_mock):
        """ """
        # Setup
        filename = 'path to data for the model'
        shuffle = False

        load_mock.return_value = {
            'f00': np.zeros((1, 5)),
            'f01': np.ones((1, 5)),
            'info': 'Metadata'
        }
        info = {
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
        json_mock.return_value = info

        expected_data = [(
            np.array([0, 0, 0, 0, 0], dtype=np.int32),
            np.array([1.]),
            np.array([1., 1., 1., 1.])
        )]

        # Run
        result = NpDataFlow(filename, shuffle)

        # Check
        assert result.shuffle is False
        assert result.info == info
        assert result.num_features == 1
        [assert_equal(actual, expected) for actual, expected in zip(result.data, expected_data)]

        load_mock.assert_called_once_with('path to data for the model')
        json_mock.assert_called_once_with('Metadata')

    @patch('tgan.np_data_flow.json.loads', autospec=True)
    @patch('tgan.np_data_flow.np.load', autospec=True)
    def test_size(self, load_mock, json_mock):
        """size return the size of self.data."""
        # Setup
        filename = 'path to data for the model'
        shuffle = False

        load_mock.return_value = {
            'info': 'Metadata'
        }

        json_mock.return_value = {
            'details': [],
            'num_features': 1,
        }

        instance = NpDataFlow(filename, shuffle)
        instance.data = [1, 2, 3]

        # Run
        result = instance.size()

        # Check
        assert result == 3
        load_mock.assert_called_once_with('path to data for the model')
        json_mock.assert_called_once_with('Metadata')

    @patch('tgan.np_data_flow.json.loads', autospec=True)
    @patch('tgan.np_data_flow.np.load', autospec=True)
    def test_get_data(self, load_mock, json_mock):
        """get_data return a generator yielding the contents of self.data."""
        # Setup
        filename = 'path to data for the model'
        shuffle = False

        load_mock.return_value = {
            'info': 'Metadata'
        }

        json_mock.return_value = {
            'details': [],
            'num_features': 1,
        }

        instance = NpDataFlow(filename, shuffle)
        instance.data = range(5)

        # Run
        generator = instance.get_data()
        result = [item for item in generator]

        # Check
        assert result == [0, 1, 2, 3, 4]

        load_mock.assert_called_once_with('path to data for the model')
        json_mock.assert_called_once_with('Metadata')

    @patch('tgan.np_data_flow.json.loads', autospec=True)
    @patch('tgan.np_data_flow.np.load', autospec=True)
    def test_get_data_shuffle(self, load_mock, json_mock):
        """get_data shuffles data when self.suffle is True.

        FIXME: It will be better to mock the whole rng attribute, however, the shuffling
        occurs inplace and I haven't found the way to do the same using mocks.
        """
        # Setup
        filename = 'path to data for the model'
        shuffle = True

        load_mock.return_value = {
            'info': 'Metadata'
        }

        json_mock.return_value = {
            'details': [],
            'num_features': 1,
        }

        instance = NpDataFlow(filename, shuffle)
        instance.data = range(5)

        instance.rng = np.random.RandomState(0)
        # Run
        generator = instance.get_data()
        result = [item for item in generator]

        # Check
        load_mock.assert_called_once_with('path to data for the model')
        json_mock.assert_called_once_with('Metadata')

        assert result == [2, 0, 1, 3, 4]
