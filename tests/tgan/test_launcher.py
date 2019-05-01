'''
from unittest import TestCase, skip
from unittest.mock import MagicMock, patch

# from tgan import launcher

launcher = None

@skip()
class TestLauncher(TestCase):

    @patch('tgan.launcher.subprocess.call', autospec=True)
    @patch('tgan.launcher.multiprocessing.current_process', autospec=True)
    def test_worker(self, current_mock, call_mock):
        """ """
        # Setup
        task_tuple = [
            ['path to executable', '--args']
        ]
        current_mock.return_value = MagicMock(_identity=[1])

        # Run
        result = launcher.worker(task_tuple)

        # Check
        assert result is None
        current_mock.assert_called_once_with()
        call_mock.assert_called_once_with(['path to executable', '--args', '--gpu', '1'])

    @patch('tgan.launcher.evaluate_classification', autospec=True)
    @patch('tgan.launcher.npz_to_csv', autospec=True)
    def test_evaluate_worker(self, npz_mock, evaluate_mock):
        """ """
        # Setup
        task_tuple = (
            'model_id',
            'model_arg',
            'epoch_id',
            'epoch_t',
            'working_dir',
            'test_csv',
            'continuous_cols'
        )

        evaluate_mock.return_value = 'score'

        expected_result = ('model_id', 'epoch_id', 'score')

        # Run
        result = launcher.evaluate_worker(task_tuple)

        # Check
        assert result == expected_result

        npz_mock.assert_called_once_with(
            'working_dir/syntheticmodel_id_epoch_t.npz',
            'working_dir/syntheticmodel_id_epoch_t.csv',
        )

        evaluate_mock.assert_called_once_with(
            'working_dir/syntheticmodel_id_epoch_t.csv', 'test_csv', 'continuous_cols')

    @patch('tgan.launcher.evaluate_classification', autospec=True)
    @patch('tgan.launcher.npz_to_csv', autospec=True)
    def test_evaluate_worker_returns_minus_1_on_error(self, npz_mock, evaluate_mock):
        """evaluate_worker returns -1 if there is an error during the scoring of model."""
        # Setup
        task_tuple = (
            'model_id',
            'model_arg',
            'epoch_id',
            'epoch_t',
            'working_dir',
            'test_csv',
            'continuous_cols'
        )

        evaluate_mock.side_effect = Exception('Something failed')

        expected_result = ('model_id', 'epoch_id', -1)

        # Run
        result = launcher.evaluate_worker(task_tuple)

        # Check
        assert result == expected_result

        npz_mock.assert_called_once_with(
            'working_dir/syntheticmodel_id_epoch_t.npz',
            'working_dir/syntheticmodel_id_epoch_t.csv',
        )

        evaluate_mock.assert_called_once_with(
            'working_dir/syntheticmodel_id_epoch_t.csv', 'test_csv', 'continuous_cols')

    @patch('tgan.launcher.LOGGER.error', autospec=True)
    @patch('tgan.launcher.os.mkdir', autospec=True)
    def test_run_experiment_wrong_folder_return_none(self, mkdir_mock, log_mock):
        """ """
        # Setup
        task = {
            'name': 'name',
            'epoch': '',
            'steps_per_epoch': '',
            'output_epoch': '',
            'sample_rows': '',
            'train_csv': '',
            'continuous_cols': ''
        }

        mkdir_mock.side_effect = Exception('something went wrong')

        # Run
        result = launcher.run_experiment(task)

        # Check
        assert result is None
        mkdir_mock.assert_called_once_with('expdir/name')
        log_mock.assert_called_once_with('skip %s, folder exist.', 'name')

'''
