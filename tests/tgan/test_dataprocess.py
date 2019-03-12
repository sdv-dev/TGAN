from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal

from tgan import dataprocess


class TestDataProcess(TestCase):

    @patch('tgan.dataprocess.np.random.rand', autospec=True)
    @patch('tgan.dataprocess.pd.DataFrame.to_csv', autospec=True)
    @patch('tgan.dataprocess.pd.read_csv', autospec=True)
    def test_split_csv(self, read_mock, csv_mock, rand_mock):
        """Split a csv file in two and save the parts."""
        # Setup
        csv_filename = 'path to csv'
        csv_out1 = 'path to split csv 1'
        csv_out2 = 'path to split csv 2'
        ratio = 0.5

        read_mock.return_value = pd.DataFrame(index=range(5), columns=list('ABC'))
        rand_mock.return_value = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        kwargs = {
            'header': False,
            'index': False
        }
        expected_csv_mock_call_args_list = [
            ((ANY, 'path to split csv 1',), kwargs),
            ((ANY, 'path to split csv 2',), kwargs)
        ]

        # Run
        result = dataprocess.split_csv(csv_filename, csv_out1, csv_out2, ratio)

        # Check
        assert result is None

        read_mock.assert_called_once_with('path to csv', header=-1)
        rand_mock.assert_called_once_with(5)
        assert csv_mock.call_args_list == expected_csv_mock_call_args_list

    @patch('tgan.dataprocess.GaussianMixture', autospec=True)
    def test_value_clustering(self, gaussian_mock):
        """Value clustering."""
        # Setup
        data = np.array([
            [0.1],
            [0.5],
            [1.0]
        ])
        n = 2

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
        result = dataprocess.value_clustering(data, n)
        features, probabilities, means, stds = result

        # Check
        assert_equal(features, expected_features)
        assert_equal(probabilities, expected_probabilities)
        assert_equal(means, expected_means)
        assert_equal(stds, expected_stds)

        gaussian_mock.assert_called_once_with(2)
        model_mock.fit.assert_called_once_with(data)
        model_mock.predict_proba.assert_called_once_with(data)

    @patch('tgan.dataprocess.np.savez', autospec=True)
    @patch('tgan.dataprocess.json.dumps', autospec=True)
    @patch('tgan.dataprocess.value_clustering', autospec=True)
    @patch('tgan.dataprocess.pd.read_csv', autospec=True)
    def test_csv_to_npz_categorical_column(self, read_mock, value_mock, json_mock, savez_mock):
        """When a column is categorical its values are mapped to integers."""
        # Setup
        csv_filename = 'Path to load the csv from.'
        npz_filename = 'Path to save the numpy arrays.'
        continuous_cols = []

        read_mock.return_value = pd.DataFrame(['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'])
        json_mock.return_value = 'string with a JSON'

        expected_data = {
            'f00': np.array([[0], [1], [2], [3], [0], [1], [2], [3]])
        }
        expected_info = {
            'num_features': 1,
            'details': [{
                'type': 'category',
                'mapping': ['A', 'B', 'C', 'D'],
                'n': 4
            }]
        }

        # Run
        result = dataprocess.csv_to_npz(csv_filename, npz_filename, continuous_cols)
        data, info = result

        # Check
        assert set(data.keys()) == set(expected_data.keys())
        assert len(data.keys()) == 1
        assert_equal(data['f00'], expected_data['f00'])

        assert info == expected_info

        read_mock.assert_called_once_with('Path to load the csv from.', header=-1)
        savez_mock.assert_called_once_with(
            'Path to save the numpy arrays.', info='string with a JSON', **data)
        json_mock.assert_called_once_with(expected_info)
        assert value_mock.call_args_list == []

    @patch('tgan.dataprocess.np.savez', autospec=True)
    @patch('tgan.dataprocess.json.dumps', autospec=True)
    @patch('tgan.dataprocess.value_clustering', autospec=True)
    @patch('tgan.dataprocess.pd.read_csv', autospec=True)
    def test_csv_to_npz_value_column(self, read_mock, value_mock, json_mock, savez_mock):
        """When a column is continous its values are clustered."""
        # Setup
        csv_filename = 'Path to load the csv from.'
        npz_filename = 'Path to save the numpy arrays.'
        continuous_cols = [0]

        read_mock.return_value = pd.DataFrame(list(range(5)))
        value_mock.return_value = (
            np.array([['feature_1'], ['feature_2']]),
            np.array([['prob_1'], ['prob_2']]),
            'means returned by value_clustering',
            'stds returned by value_clustering',
        )
        json_mock.return_value = 'string with a JSON'

        expected_data = {
            'f00': np.array([
                ['feature_1', 'prob_1'],
                ['feature_2', 'prob_2']
            ])
        }
        expected_info = {
            'num_features': 1,
            'details': [{
                'type': 'value',
                'means': 'means returned by value_clustering',
                'stds': 'stds returned by value_clustering',
                'n': 5
            }]
        }

        # Run
        result = dataprocess.csv_to_npz(csv_filename, npz_filename, continuous_cols)
        data, info = result

        # Check
        # Mock asserts don't work fine with numpy.arrays.
        assert set(data.keys()) == set(expected_data.keys())
        assert len(data.keys()) == 1
        assert_equal(data['f00'], expected_data['f00'])

        assert info == expected_info

        read_mock.assert_called_once_with('Path to load the csv from.', header=-1)
        json_mock.assert_called_once_with(expected_info)
        savez_mock.assert_called_once_with(
            'Path to save the numpy arrays.', info='string with a JSON', **data)

        assert len(value_mock.call_args_list) == 1
        call_args, call_kwargs = value_mock.call_args_list[0]  # tuple with args and kwargs

        assert len(call_args) == 2
        assert_equal(call_args[0], np.array([[0], [1], [2], [3], [4]]))
        assert call_args[1] == 5
        assert call_kwargs == {}

    def test__rev_feature_raises_value_error_unsupported_type(self):
        """_rev_feature will raise a ValueError if info['type'] is not supported."""
        # Setup
        data = None
        info = {
            'type': 'invalid_type'
        }

        expected_error_message = (
            "info['type'] must be either 'category' or 'value'.Instead it was 'invalid_type'.")

        try:
            # Run
            dataprocess._rev_feature(data, info)
        except ValueError as error:
            # Check
            assert len(error.args) == 1
            assert error.args[0] == expected_error_message

    def test__rev_feature_categorical_column_raise_value_error_invalid_format(self):
        """_rev_feature will raise an exception for categorical columns with invalid shape."""
        # Setup
        data = np.array([])
        info = {
            'type': 'category'
        }

        expected_error_message = 'The argument `data` must be a numpy.ndarray with shape (n, 1).'

        try:
            # Run
            dataprocess._rev_feature(data, info)
        except ValueError as error:
            # Check
            assert len(error.args) == 1
            assert error.args[0] == expected_error_message

    def test__rev_feature_categorical_column(self):
        """_rev_feature maps categorical values back to their original values."""
        # Setup
        data = np.array([[0], [1], [2], [1], [0]])
        info = {
            'type': 'category',
            'mapping': ['A', 'B', 'C']
        }

        expected_result = ['A', 'B', 'C', 'B', 'A']

        # Run
        result = dataprocess._rev_feature(data, info)

        # Check
        assert result == expected_result

    def test__rev_feature_continous_column(self):
        """_rev_feature reverses the process of cluster_values on continuous columns."""
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

        expected_result = np.array([0.1, 0.5, 1.0])

        # Run
        result = dataprocess._rev_feature(data, info)

        # Check
        assert_allclose(result, expected_result)  # Minor difference due to floating point.

    @patch('tgan.dataprocess.np.load', autospec=True)
    @patch('tgan.dataprocess.json.loads', autospec=True)
    @patch('tgan.dataprocess._rev_feature', autospec=True)
    @patch('tgan.dataprocess.pd.DataFrame.to_csv', autospec=True)
    def test_npz_to_csv(self, csv_mock, rev_mock, json_mock, load_mock):
        """ """
        # Setup
        npfilename = 'Path to the origin npz file.'
        csvfilename = 'Path to the destination csv file'

        load_mock.return_value = {
            'info': 'Metadata for the given array.',
            'f00': np.array(['stored', 'values'])
        }

        json_mock.return_value = {
            'num_features': 1,
            'details': [{
                'type': 'value',
                'means': 'means returned by value_clustering',
                'stds': 'stds returned by value_clustering',
                'n': 5
            }]
        }

        rev_mock.return_value = np.array(['result of', '_rev_feature'])

        expected_result = pd.DataFrame(['result of', '_rev_feature'])

        # Run
        result = dataprocess.npz_to_csv(npfilename, csvfilename)

        # Check
        assert result.equals(expected_result)

        load_mock.assert_called_once_with('Path to the origin npz file.')
        json_mock.assert_called_once_with('Metadata for the given array.')
        csv_mock.assert_called_once_with(
            result, 'Path to the destination csv file', index=False, header=False)

        assert len(rev_mock.call_args_list) == 1
        call_args, call_kwargs = rev_mock.call_args_list[0]  # tuple with args and kwargs

        assert len(call_args) == 2
        assert_equal(call_args[0], np.array(['stored', 'values']))
        assert call_args[1] == {
            'type': 'value',
            'means': 'means returned by value_clustering',
            'stds': 'stds returned by value_clustering',
            'n': 5
        }
        assert call_kwargs == {}
