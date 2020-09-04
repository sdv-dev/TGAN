<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="sdv-dev" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/TGAN.svg)](https://pypi.python.org/pypi/TGAN)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/TGAN.svg?branch=master)](https://travis-ci.org/sdv-dev/TGAN)
[![CodeCov](https://codecov.io/gh/sdv-dev/TGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/TGAN)
[![Downloads](https://pepy.tech/badge/tgan)](https://pepy.tech/project/tgan)

__We are happy to announce that our new model for synthetic data called [CTGAN](https://github.com/sdv-dev/CTGAN) is open-sourced. Please check the new model in [this repo](https://github.com/sdv-dev/CTGAN). The new model is simpler and gives better performance on many datasets.__

# TGAN

Generative adversarial training for synthesizing tabular data.

* License: [MIT](https://github.com/sdv-dev/TGAN/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/sdv-dev/TGAN

# Overview

TGAN is a tabular data synthesizer. It can generate fully synthetic data from real data. Currently, TGAN can
generate numerical columns and categorical columns.

# Requirements

## Python

**TGAN** has been developed and runs on Python [3.5](https://www.python.org/downloads/release/python-356/),
[3.6](https://www.python.org/downloads/release/python-360/) and
[3.7](https://www.python.org/downloads/release/python-370/).

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system where **TGAN**
is run.

# Installation

The simplest and recommended way to install TGAN is using `pip`:

```
pip install tgan
```

Alternatively, you can also clone the repository and install it from sources

```
git clone git@github.com:sdv-dev/TGAN.git
cd TGAN
make install
```

For development, you can use `make install-develop` instead in order to install all the required
dependencies for testing and code linting.

# Data Format

## Input Format

In order to be able to sample new synthetic data, **TGAN** first needs to be *fitted* to
existing data.

The input data for this *fitting* process has to be a single table that satisfies the following
rules:

* Has no missing values.
* Has columns of types `int`, `float`, `str` or `bool`.
* Each column contains data of only one type.

An example of such a tables would be:

| str_column | float_column | int_column | bool_column |
|------------|--------------|------------|-------------|
|    'green' |         0.15 |         10 |        True |
|     'blue' |         7.25 |         23 |       False |
|      'red' |        10.00 |          1 |       False |
|   'yellow' |         5.50 |         17 |        True |

As you can see, this table contains 4 columns: `str_column`, `float_column`, `int_column` and
`bool_column`, each one being an example of the supported value types. Notice aswell that there is
no missing values for any of the rows.

**NOTE**: It's important to have properly identifed which of the columns are numerical, which means
that they represent a magnitude, and which ones are categorical, as during the preprocessing of
the data, numerical and categorical columns will be processed differently.

## Output Format

The output of **TGAN** is a table of sampled data with the same columns as the input table and as
many rows as requested.

## Demo Datasets

**TGAN** includes a few datasets to use for development or demonstration purposes. These datasets
come from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml), and have been
preprocessed to be ready to use with **TGAN**, following the requirements specified in the
[Input Format](#input-format) section.

These datasets can be browsed and directly downloaded from the
[hdi-project-tgan AWS S3 Bucket](http://hdi-project-tgan.s3.amazonaws.com/index.html)

### Census dataset

This dataset contains a single table, with information from the census, labeled with information of
wheter or not the income of is greater than 50.000 $/year. It's a single csv file, containing
199522 rows and 41 columns. From these 41 columns, only 7 are identified as continuous. In
**TGAN** this dataset is called `census`.

### Cover type

This dataset contains a single table with cartographic information labeled with the different
forrest cover types. It's a single csv file, containing 465588 rows and 55 columns. From these
55 columns, 10 are identified as continuous. In **TGAN** this dataset is called `covertype`.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with the most basic usage of **TGAN** in order to generate samples from a given dataset.

**NOTE**: The following examples are also covered in a [Jupyter](https://jupyter.org/) notebook,
which you can execute by running the following commands inside your *virtualenv*:

```
pip install jupyter
jupyter notebook examples/Usage_Example.ipynb
```

## 1. Load the data

The first step is to load the data wich we will use to fit TGAN. In order to do so, we will first
import the function `tgan.data.load_data` and call it with the name of the dataset that we want to
load.

In this case, we will load the `census` dataset, which we will use during the subsequent steps,
and obtain two objects:

1. `data`, that will contain a `pandas.DataFrame` with the table of data from the `census`
dataset ready to be used to fit the model.

2. `continuous_columns`, that will contain a `list` with the indices of continuous columns.

```
>>> from tgan.data import load_demo_data
>>> data, continuous_columns = load_demo_data('census')
>>> data.head(3).T[:10]
                              0                                     1                             2
0                            73                                    58                            18
1               Not in universe        Self-employed-not incorporated               Not in universe
2                             0                                     4                             0
3                             0                                    34                             0
4          High school graduate            Some college but no degree                    10th grade
5                             0                                     0                             0
6               Not in universe                       Not in universe                   High school
7                       Widowed                              Divorced                 Never married
8   Not in universe or children                          Construction   Not in universe or children
9               Not in universe   Precision production craft & repair               Not in universe

>>> continuous_columns
[0, 5, 16, 17, 18, 29, 38]

```

## 2. Create a TGAN instance

The next step is to import TGAN and create an instance of the model.

To do so, we need to import the `tgan.model.TGANModel` class and call it with the
`continuous_columns` as unique argument.

This will create a TGAN instance with the default parameters:

```
>>> from tgan.model import TGANModel
>>> tgan = TGANModel(continuous_columns)
```

## 3. Fit the model

Once you have a **TGAN** instance, you can proceed to call it's `fit` method passing the `data` that
you loaded before in order to start the fitting process:

```
>>> tgan.fit(data)
```

This process will not return anything, however, the progress of the fitting will be printed in the
screen.

**NOTE** Depending on the performance of the system you are running, and the parameters selected
for the model, this step can take up to a few hours.

## 4. Sample new data

After the model has been fitted, you are ready to generate new samples by calling the `sample`
method of the `TGAN` instance passing it the desired amount of samples:

```
>>> num_samples = 1000
>>> samples = tgan.sample(num_samples)
>>> samples.head(3).T[:10]
                                         0                                     1                                   2
0                                       12                                    27                                  56

0                                       12                                    27                                  56
1                          Not in universe        Self-employed-not incorporated                             Private
2                                        0                                     4                                  35
3                                        0                                    34                                  22
4                                 Children            Some college but no degree          Some college but no degree
5                                        0                                     0                                 500
6                          Not in universe                       Not in universe                     Not in universe
7                            Never married       Married-civilian spouse present     Married-civilian spouse present
8              Not in universe or children                          Construction   Finance insurance and real estate
9                          Not in universe   Precision production craft & repair      Adm support including clerical

```

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with
the same format as the input data and 1000 rows as we requested.

## 5. Save and Load a model

In the steps above we saw that the fitting process can take a lot of time, so we probably would
like to avoid having to fit every we want to generate samples. Instead we can fit a model once,
save it, and load it every time we want to sample new data.

If we have a fitted model, we can save it by calling it's `save` method, that only takes
as argument the path where the model will be stored. Similarly, the `TGANModel.load` allows to load
a model stored on disk by passing as argument the path where the model is stored.

```
>>> model_path = 'models/mymodel.pkl'
>>> tgan.save(model_path)
Model saved successfully.
```

Bear in mind that in case the file already exists, **TGAN** will avoid overwritting it unless the
`force=True` argument is passed:

```
>>> tgan.save(model_path)
The indicated path already exists. Use `force=True` to overwrite.
```

In order to do so:

```
>>> tgan.save(model_path, force=True)
Model saved successfully.
```

Once the model is saved, it can be loaded back as a **TGAN** instance by using the `TGANModel.load`
method:

```
>>> new_tgan = TGANModel.load(model_path)
>>> new_samples = new_tgan.sample(num_samples)
>>> new_samples.head(3).T[:10]

                                         0                                     1                                   2
0                                       12                                    27                                  56

0                                       12                                    27                                  56
1                          Not in universe        Self-employed-not incorporated                             Private
2                                        0                                     4                                  35
3                                        0                                    34                                  22
4                                 Children            Some college but no degree          Some college but no degree
5                                        0                                     0                                 500
6                          Not in universe                       Not in universe                     Not in universe
7                            Never married       Married-civilian spouse present     Married-civilian spouse present
8              Not in universe or children                          Construction   Finance insurance and real estate
9                          Not in universe   Precision production craft & repair      Adm support including clerical
```

At this point we could use this model instance to generate more samples.

# Loading custom datasets

In the previous steps we used some demonstration data but we did not show you how to load your own
dataset.

In order to do so you will need to generate a `pandas.DataFrame` object from your dataset. If your
dataset is in a `csv` format you can do so by using `pandas.read_csv` and passing to it the path to
the CSV file that you want to load.

Additionally, you will need to create 0-indexed list of columns indices to be considered continuous.

For example, if we want to load a local CSV file, `path/to/my.csv`, that has as continuous columns
their first 4 columns, that is, indices `[0, 1, 2, 3]`, we would do it like this:

```
>>> import pandas as pd
>>> data = pd.read_csv('data/census.csv')
>>> continuous_columns = [0, 1, 2, 3]
```

Now you can use the `continuous_columns` to create a **TGAN** instance and use the `data` to `fit`
it, like we did before:

```
>>> from tgan.model import TGANModel
>>> tgan = TGANModel(continuous_columns)
>>> tgan.fit(data)
```

# Model Parameters

If you want to change the default behavior of `TGANModel`, such as as different `batch_size` or
`num_epochs`, you can do so by passing different arguments when creating the instance.

## Model general behavior

* continous_columns (`list[int]`, required): List of columns indices to be considered continuous.
* output (`str`, default=`output`): Path to store the model and its artifacts.

## Neural network definition and fitting

* max_epoch (`int`, default=`100`): Number of epochs to use during training.
* steps_per_epoch (`int`, default=`10000`): Number of steps to run on each epoch.
* save_checkpoints(`bool`, default=True): Whether or not to store checkpoints of the model after each training epoch.
* restore_session(`bool`, default=True): Whether or not continue training from the last checkpoint.
* batch_size (`int`, default=`200`): Size of the batch to feed the model at each step.
* z_dim (`int`, default=`100`): Number of dimensions in the noise input for the generator.
* noise (`float`, default=`0.2`): Upper bound to the gaussian noise added to categorical columns.
* l2norm (`float`, default=`0.00001`): L2 reguralization coefficient when computing losses.
* learning_rate (`float`, default=`0.001`): Learning rate for the optimizer.
* num_gen_rnn (`int`, default=`400`): Number of units in rnn cell in generator.
* num_gen_feature (`int`, default=`100`): Number of units in fully connected layer in generator.
* num_dis_layers (`int`, default=`2`): Number of layers in discriminator.
* num_dis_hidden (`int`, default=`200`): Number of units per layer in discriminator.
* optimizer (`str`, default=`AdamOptimizer`): Name of the optimizer to use during `fit`, possible
  values are: [`GradientDescentOptimizer`, `AdamOptimizer`, `AdadeltaOptimizer`].

If you wanted to create an identical instance to the one created on step 2, but passing the
arguments in a explicit way, this can be achieved with the following lines:

```
>>> from tgan.model import TGANModel
>>> tgan = TGANModel(
   ...:     continuous_columns,
   ...:     output='output',
   ...:     max_epoch=5,
   ...:     steps_per_epoch=10000,
   ...:     save_checkpoints=True,
   ...:     restore_session=True,
   ...:     batch_size=200,
   ...:     z_dim=200,
   ...:     noise=0.2,
   ...:     l2norm=0.00001,
   ...:     learning_rate=0.001,
   ...:     num_gen_rnn=100,
   ...:     num_gen_feature=100,
   ...:     num_dis_layers=1,
   ...:     num_dis_hidden=100,
   ...:     optimizer='AdamOptimizer'
   ...: )
```

# Command-line interface

We include a command-line interface that allows users to access TGAN functionality. Currently only
one action is supported.

## Random hyperparameter search

### Input

To run random searchs for the best model hyperparameters for a given dataset, we will need:

* A dataset, in a csv file, without any missing value, only columns of type `bool`, `str`, `int` or
  `float` and only one type for column, as specified in the [Input Format](#input-format).

* A JSON file containing the configuration for the search. This configuration shall contain:

  * `name`: Name of the experiment. A folder with this name will be created.
  * `num_random_search`: Number of iterations in hyper parameter search.
  * `train_csv`: Path to the csv file containing the dataset.
  * `continuous_cols`: List of column indices, starting at 0, to be considered continuous.
  * `epoch`: Number of epoches to train the model.
  * `steps_per_epoch`: Number of optimization steps in each epoch.
  * `sample_rows`: Number of rows to sample when evaluating the model.

You can see an example of such a json file in [examples/config.json](examples/config.json), which you
can download and use as a template.

### Execution

Once we have prepared everything we can launch the random hyperparameter search with this command:

``` bash
tgan experiments config.json results.json
```

Where the first argument, `config.json`, is the path to your configuration JSON, and the second,
`results.json`, is the path to store the summary of the execution.

This will run the random search, wich basically consist of the folling steps:

1. We fetch and split our data between test and train.
2. We randomly select the hyperparameters to test.
3. Then, for each hyperparameter combination, we train a TGAN model using the real training data T
   and generate a synthetic training dataset Tsynth.
4. We then train machine learning models on both the real and synthetic datasets.
5. We use these trained models on real test data and see how well they perform.

### Output

After the experiment has finished, the following can be found:

* A JSON file, in the example above called `results.json`, containing a summary of the experiments.
  This JSON will contain a key for each experiment `name`, and on it, an array of length
  `num_random_search`, with the selected parameters and its evaluation score. For a configuration
  like the example, the summary will look like this:

``` python
{
    'census': [
        {
            "steps_per_epoch" : 10000,
            "num_gen_feature" : 300,
            "num_dis_hidden" : 300,
            "batch_size" : 100,
            "num_gen_rnn" : 400,
            "score" : 0.937802280415988,
            "max_epoch" : 5,
            "num_dis_layers" : 4,
            "learning_rate" : 0.0002,
            "z_dim" : 100,
            "noise" : 0.2
        },
        ... # 9 more nodes
    ]
}
```

* A set of folders, each one names after the `name` specified in the JSON configuration, contained
in the `experiments` folder. In each folder, sampled data and the models can be found. For a configuration
like the example, this will look like this:

```
experiments/
  census/
    data/       # Sampled data with each of the models in the random search.
    model_0/
      logs/     # Training logs
      model/    # Tensorflow model checkpoints
    model_1/    # 9 more folders, one for each model in the random search
    ...
```

# Research

The first **TGAN** version was built as the supporting software for the [Synthesizing Tabular Data using Generative Adversarial Networks](https://arxiv.org/pdf/1811.11264.pdf) paper by Lei Xu and Kalyan Veeramachaneni.

The exact version of software mentioned in the paper can be found in the releases section as the [research pre-release](https://github.com/sdv-dev/TGAN/releases/tag/research)

# Citing TGAN

If you use TGAN for yor research, please consider citing the following paper (https://arxiv.org/pdf/1811.11264.pdf):

If you use TGAN, please cite the following work:

> Lei Xu, Kalyan Veeramachaneni. 2018. Synthesizing Tabular Data using Generative Adversarial Networks.

```LaTeX
@article{xu2018synthesizing,
  title={Synthesizing Tabular Data using Generative Adversarial Networks},
  author={Xu, Lei and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1811.11264},
  year={2018}
}
```
