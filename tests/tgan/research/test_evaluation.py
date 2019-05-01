from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_equal

from tgan.research import evaluation


class TestEvaluation(TestCase):

    def test__proc_data_continous_column(self):
        """_proc_data adds a dimension in columns with continuous values."""
        # Setup
        df = pd.DataFrame({
            0: [1, 2, 3],
            1: ['label_1', 'label_2', 'label_3']
        })
        continuous_cols = [0]

        expected_features = np.array([[1], [2], [3]])
        expected_labels = np.array(['label_1', 'label_2', 'label_3'])

        # Run
        result = evaluation._proc_data(df, continuous_cols)
        features, labels = result

        # Check
        assert_equal(features, expected_features)
        assert_equal(labels, expected_labels)

    def test__proc_data_category_column(self):
        """_proc_data One-hot encodes categorical columns."""
        # Setup
        df = pd.DataFrame({
            0: ['A', 'B', 'A'],
            1: ['label_1', 'label_2', 'label_3']
        })
        continuous_cols = []

        expected_features = np.array([[1, 0], [0, 1], [1, 0]])
        expected_labels = np.array(['label_1', 'label_2', 'label_3'])

        # Run
        result = evaluation._proc_data(df, continuous_cols)
        features, labels = result

        # Check
        assert_equal(features, expected_features)
        assert_equal(labels, expected_labels)

    @patch('tgan.research.evaluation._proc_data', autospec=True)
    def test_evaluate_classification(self, proc_mock):
        """ """
        # Setup
        train_csv = pd.DataFrame(['train_data'])
        test_csv = pd.DataFrame(['test_data'])
        continuous_cols = []
        classifier_spec = {'predict.return_value': 'array of predictions'}
        classifier = MagicMock(**classifier_spec)
        metric = MagicMock(return_value='score for model')

        proc_mock.return_value = (['feature_1', 'feature_2'], ['label_1', 'label_2'])

        expected_result = 'score for model'

        # Run
        result = evaluation.evaluate_classification(
            train_csv, test_csv, continuous_cols, classifier, metric)

        # Check
        assert result == expected_result

        classifier.fit.assert_called_once_with(['feature_1'], ['label_1'])
        classifier.predict.assert_called_once_with(['feature_2'])
        metric.assert_called_once_with(['label_2'], 'array of predictions')
