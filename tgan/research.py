"""Tune and evaluate TGAN models."""

import numpy as np
from sklearn.model_selection import train_test_split

from tgan.data import load_data
from tgan.model import TUNABLE_VARIABLES, TGANModel

# from tgan.evaluation import evaluate_classification


def prepare_hyperparameter_search(steps_per_epoch, num_random_search):
    """Prepare hyperparameters."""
    model_kwargs = []
    basic_kwargs = {
        'max_epoch': 1,
        'steps_per_epoch': steps_per_epoch,
        'store_checkpoints': True
    }
    for i in range(num_random_search):
        kwargs = {name: np.random.choice(choices) for name, choices in TUNABLE_VARIABLES.items()}
        kwargs.update(basic_kwargs)
        model_kwargs.append(kwargs)

    return model_kwargs


def run_experiment(
    name, max_epoch, steps_per_epoch, output_epoch, sample_rows, file_path, continuous_columns,
    num_random_search
):
    """Run experiment using the given params and collect the results.

    The experiment run the following steps:
    1. We fetch and split our data between test and train
    2. We first train a TGAN data synthesizer using the real training data T and generate a
       synthetic training dataset Tsynth.
    3. We then train machine learning models on both the real and synthetic datasets.
    4. We use these trained models on real test data and see how well they perform.

    """
    # Load and split data
    data = load_data(file_path)
    test_index, train_index = train_test_split(data, train_size=0.8)

    # Prepare hyperparameter search
    model_kwargs = prepare_hyperparameter_search(steps_per_epoch, num_random_search)

    synthesized_data = {}

    # Training models and sampling data
    for index, kwargs in enumerate(model_kwargs):
        for epoch in range(max_epoch):
            model = TGANModel(**kwargs, output='{}}/model_{}'.format(name, index))
            model.fit(data)
            synthesized_data[(index, epoch)] = model.sample(sample_rows)
    # Evaluating synthesized data
