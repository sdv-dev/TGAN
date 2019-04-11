#!usr/bin/env python

"""Tune and evaluate TGAN models."""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorpack.utils import logger

from tgan.model import TUNABLE_VARIABLES, TGANModel
from tgan.research.evaluation import evaluate_classification


def prepare_hyperparameter_search(epoch, steps_per_epoch, num_random_search):
    """Prepare hyperparameters."""
    model_kwargs = []
    basic_kwargs = {
        'max_epoch': epoch,
        'steps_per_epoch': steps_per_epoch,
    }

    for i in range(num_random_search):
        kwargs = {name: np.random.choice(choices) for name, choices in TUNABLE_VARIABLES.items()}
        kwargs.update(basic_kwargs)
        model_kwargs.append(kwargs)

    return model_kwargs


def fit_score_model(
        name, model_kwargs, train_data, test_data, continuous_columns,
        sample_rows, store_samples
):
    """Fit and score models using given params."""
    for index, kwargs in enumerate(model_kwargs):
        logger.info('Training TGAN Model %d/%d', index + 1, len(model_kwargs))

        tf.reset_default_graph()
        base_dir = os.path.join('experiments', name)
        output = os.path.join(base_dir, 'model_{}'.format(index))
        model = TGANModel(continuous_columns, output=output, **kwargs)
        model.fit(train_data)
        sampled_data = model.sample(sample_rows)

        if store_samples:
            dir_name = os.path.join(base_dir, 'data')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            file_name = os.path.join(dir_name, 'model_{}.csv'.format(index))
            sampled_data.to_csv(file_name, index=False, header=True)

        score = evaluate_classification(sampled_data, test_data, continuous_columns)
        model_kwargs[index]['score'] = score

    return model_kwargs


def run_experiment(
    name, epoch, steps_per_epoch, sample_rows, train_csv, continuous_cols,
    num_random_search, store_samples=True, force=False
):
    """Run experiment using the given params and collect the results.

    The experiment run the following steps:

    1. We fetch and split our data between test and train.

    2. We first train a TGAN data synthesizer using the real training data T and generate a
       synthetic training dataset Tsynth.

    3. We then train machine learning models on both the real and synthetic datasets.

    4. We use these trained models on real test data and see how well they perform.

    """
    if os.path.isdir(name):
        if force:
            logger.info('Folder "{}" exists, and force=True. Deleting folder.'.format(name))
            os.rmdir(name)

        else:
            raise ValueError(
                'Folder "{}" already exist. Please, use force=True to force deletion '
                'or use a different name.'.format(name))

    # Load and split data
    data = pd.read_csv(train_csv, header=-1)
    train_data, test_data = train_test_split(data, train_size=0.8)

    # Prepare hyperparameter search
    model_kwargs = prepare_hyperparameter_search(epoch, steps_per_epoch, num_random_search)

    return fit_score_model(
        name, model_kwargs, train_data, test_data,
        continuous_cols, sample_rows, store_samples
    )


def numpy_default(obj):
    """Change numpy objects into json-serializable ones."""
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)

    if isinstance(obj, [np.float64]):
        return float(obj)

    raise TypeError


def run_experiments(config_path, output_path):
    """Run experiments specified in JSON file."""
    with open(config_path) as f:
        experiments_config = json.load(f)

    result = {}
    if isinstance(experiments_config, list):
        for experiment in experiments_config:
            name = experiment['name']
            result[name] = run_experiment(**experiment)
    else:
        name = experiments_config['name']
        result[name] = run_experiment(**experiments_config)

    with open(output_path, 'w') as f:
        json.dump(result, f, default=numpy_default)
