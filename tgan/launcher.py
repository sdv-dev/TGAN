"""Multiprocessing utilities."""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess

import numpy as np

from tgan.dataprocess import csv_to_npz, npz_to_csv, split_csv
from tgan.evaluation import evaluate_classification
from tgan.TGAN_synthesizer import tunable_variables

TEST_DIR = 'expdir'
LOGGER = logging.getLogger(__name__)


def worker(task_tuple):
    """Invoke subprocess for each element in task_tuple.

    Args:
        task_tuple(list[list[str]]): List calls using the subprocess syntax.

    Returns:
        None

    """
    worker_id = multiprocessing.current_process()._identity[0]
    gpu_id = worker_id % 2

    for task in task_tuple:
        subprocess.call(task + ['--gpu', str(gpu_id)])


def evaluate_worker(task_tuple):
    """Evaluate the performance of the model trained in a worker.

    Args:
        task_tuple(tuple[str]): Parameters of the task (model_id, model_args, epoch_id, epoch_t,
           working_dir, test_csv, continous_cols).

    Returns:
        tuple[str, str, float]: :attr:`model_id`, :attr:`epoch_id` and the score .

    Note:
        If there is an error while evaluating the model, will return an score of **-1**

    """
    model_id, model_arg, epoch_id, epoch_t, working_dir, test_csv, continuous_cols = task_tuple

    syn_path = os.path.join(working_dir, 'synthetic{}_{}.npz'.format(model_id, epoch_t))
    csv_path = os.path.join(working_dir, 'synthetic{}_{}.csv'.format(model_id, epoch_t))
    npz_to_csv(syn_path, csv_path)

    try:
        score = evaluate_classification(csv_path, test_csv, continuous_cols)

    except Exception:
        score = -1

    return (model_id, epoch_id, score)


def run_experiment(task):
    """Run an experiment using the given values.

    Args:
        task(dict): Parameters for the experiment.

    Returns:
        None

    Note:
        The results of the experiment are stored in :attr:`expdir/task['name']/exp-result.csv`.

    """
    name = task['name']
    epoch = task['epoch']
    steps = task['steps_per_epoch']
    output_epoch = task['output_epoch']
    sample_rows = task['sample_rows']
    train_csv = task['train_csv']
    continuous_cols = task['continuous_cols']

    working_dir = os.path.join(TEST_DIR, name)
    try:
        os.mkdir(working_dir)

    except Exception:
        LOGGER.error('skip %s, folder exist.', name)
        return

    train_csv_part1 = os.path.join(working_dir, 'data_I.csv')
    train_csv_part2 = os.path.join(working_dir, 'data_II.csv')
    train_npz = os.path.join(working_dir, 'train.npz')
    split_csv(train_csv, train_csv_part1, train_csv_part2)

    csv_to_npz(train_csv_part1, train_npz, continuous_cols)
    model_gobal_param = [
        '--max_epoch', str(epoch),
        '--steps_per_epoch', str(steps),
        '--data', train_npz
    ]

    models = []
    for i in range(task['num_random_search']):
        current_model = ['src/TGAN_synthesizer.py']

        for key, choices in tunable_variables.items():
            current_model.append(key)
            current_model.append(str(np.random.choice(choices)))
        models.append(current_model)

    all_params = {
        'sample_rows': sample_rows,
        'models': models,
        'model_gobal_param': model_gobal_param
    }

    commands = []
    for model_id, model_arg in enumerate(models):
        calls = []
        call_train = ['python3'] + model_arg + \
            ['--exp_name', '{}-{}'.format(name, model_id)] + model_gobal_param
        calls.append(call_train)

        for epoch_t in range(epoch - output_epoch + 1, epoch + 1):
            syn_path = os.path.join(working_dir, 'synthetic{}_{}.npz'.format(model_id, epoch_t))

            call_test = ['python3'] + model_arg + model_gobal_param + [
                '--sample', str(sample_rows),
                '--load', 'train_log/{}:{}-{}/model-{}'.format(
                    'TGAN_synthesizer', name, model_id, epoch_t * steps
                ),
                '--output', syn_path
            ]
            calls.append(call_test)

        commands.append(calls)

    # parallel experiments
    pool = multiprocessing.Pool(2)
    pool.map(worker, commands)
    pool.close()
    pool.join()

    commands = []
    for model_id, model_arg in enumerate(models):
        for epoch_id, epoch_t in enumerate(range(epoch - output_epoch + 1, epoch + 1)):
            commands.append((
                model_id, model_arg, epoch_id, epoch_t,
                working_dir, train_csv_part2, continuous_cols
            ))

    pool = multiprocessing.Pool(2)
    evaluate_res = pool.map(evaluate_worker, commands)
    pool.close()
    pool.join()

    score_arr = np.zeros((len(models), output_epoch))

    for model_id, epoch_id, score in evaluate_res:
        score_arr[model_id, epoch_id] = score

    with open(os.path.join(working_dir, 'exp-params.json'), 'w') as f:
        json.dump(all_params, f)

    np.savetxt(os.path.join(working_dir, 'exp-result.csv'), score_arr, delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a json config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(config)

    try:
        os.mkdir(TEST_DIR)
    except Exception:
        pass

    for task in config:
        run_experiment(task)
