"""Command Line Interface for TGAN."""

import argparse

from tgan.research.experiments import run_experiments


def get_train_parser():
    """Build the ArgumentParser for CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', type=int, default=0,
                        help='the number of samples in the synthetic output.')
    parser.add_argument('--data', required=True, help='a npz file')
    parser.add_argument('--output', type=str)
    parser.add_argument('--exp_name', type=str, default=None)

    # parameters for model tuning.
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)

    parser.add_argument('--num_gen_rnn', type=int, default=400)
    parser.add_argument('--num_gen_feature', type=int, default=100)

    parser.add_argument('--num_dis_layers', type=int, default=2)
    parser.add_argument('--num_dis_hidden', type=int, default=200)

    parser.add_argument('--noise', type=float, default=0.2)

    parser.add_argument('--optimizer', type=str, default='AdamOptimizer',
                        choices=['GradientDescentOptimizer', 'AdamOptimizer', 'AdadeltaOptimizer'])
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--l2norm', type=float, default=0.00001)

    return parser


def get_parser():
    """Build argument parser for TGAN CLI utility."""
    parser = argparse.ArgumentParser(description='TGAN Command Line Interface.')
    parser.set_defaults(function=None)

    action = parser.add_subparsers(title='action', dest='action')
    action.required = True

    experiments = action.add_parser('experiments', help='Run experiments using TGAN.')
    experiments.add_argument(
        'input', type=str, help='Path to the JSON file with the configuration.')
    experiments.add_argument(
        'output', type=str, help='Path to store the results.')

    return parser


def main():
    """Python Entry point for CLI."""
    parser = get_parser()
    args = parser.parse_args()
    run_experiments(args.input, args.output)
