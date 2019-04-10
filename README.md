<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="SteganoGAN" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/TGAN.svg)](https://pypi.python.org/pypi/TGAN)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/TGAN.svg?branch=master)](https://travis-ci.org/DAI-Lab/TGAN)

# TGAN

Generative adversarial training for synthesizing tabular data.

TGAN is a tabular data synthesizer. It can generate fully synthetic data from real data. Currently, TGAN can
generate numerical columns and categorical columns. This software can do random search for TGAN parameters
on multiple datasets using multiple GPUs.

* Free software: MIT license
* Documentation: https://DAI-Lab.github.io/tgan
* Homepage: https://github.com/DAI-Lab/tgan

## Getting Started

### Requirements

#### Python

**TGAN** has been developed and runs on Python [3.5](https://www.python.org/downloads/release/python-356/),
[3.6](https://www.python.org/downloads/release/python-360/) and
[3.7](https://www.python.org/downloads/release/python-370/).

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system where **TGAN**
is run.

### Installation

The simplest and recommended way to install TGAN is using `pip`:

```
pip install tgan
```

Alternatively, you can also clone the repository and install it from sources

```
git clone git@github.com:DAI-Lab/tgan.git
cd tgan
make install
```

For development, you can use `make install-develop` instead in order to install all the required
dependencies for testing and code linting.

### Data Format

#### Input

In order to be able to sample new synthetic data, **TGAN** needs to first be *fitted* to existing data.

The input data for this *fitting* process has to be a single table that:

* Has no missing values.
* Has columns of types `int`, `float`, `str` or `bool`.
* Each column contains data of only one type.

The following is a simple example of a table with 4 columns, `str_column`, `float_column`,`int_column`,
`bool_column`, each one being an example of one of the supported value types.

| str_column | float_column | int_column | bool_column |
|------------|--------------|------------|-------------|
|    'green' |         0.15 |         10 |        True |
|     'blue' |         7.25 |         23 |       False |
|      'red' |        10.00 |          1 |       False |
|   'yellow' |         5.50 |         17 |        True |

**NOTE**: It's important to have properly identifed which of the columns are numerical, which means
that they represent a magnitude, and which ones are categorical, as during the preprocessing of
the data, numerical and categorical columns will be processed differently.

#### Output

The output of **TGAN** is a table of sampled data with the same columns as the input table and as many rows as requested.

### Demo Datasets

**TGAN** includes a few datasets to use for development or demonstration purposes. These datasets
come from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml), but have been
preprocessed to be ready to use with **TGAN**, following the requirements specified in the `Input` section.

These datasets can be browsed and directly downloaded from the
[tgan-demo AWS S3 Bucket](https://s3.amazonaws.com/hdi-demos/tgan-demo/)

#### Census dataset

This dataset contains a single table, with information from the census, labeled with information of
wheter or not the income of is greater than 50.000 $/year. It's a single csv file, containing
199522 rows and 41 columns. From these 41 columns, only 7 are identified as continuous. In **TGAN** this
dataset is called `census`.

#### Cover type

This dataset contains a single table with cartographic information labeled with the different
forrest cover types. It's a single csv file, containing 465588 rows and 55 columns. From these
55 columns, 10 are identified as continuous. In **TGAN** this dataset is called `covertype`.

## Quickstart

In this section we will show the most basic usage of **TGAN** in order to generate samples from a
given dataset.

**NOTE**: All the examples of this tutorial are run in an [IPython Shell](https://ipython.org/),
which you can install by running the following commands inside your *virtualenv*:

```
pip install ipython
ipython
```

### 1. Load the data

The first step is to load the data wich we will use to fit TGAN. In order to do so, we will first
import the function `tgan.data.load_data` and call it with the name the dataset that we want to
load.

In this case, we will load the `census` dataset, which we will use during the subsequent steps,
and obtain two objects:

1. `data` will contain a `pandas.DataFrame` with the table of data from the `census` dataset ready
   to be used to fit the model.

2. `continous_columns` will contain a `list` with the indices of continuous columns.

```
In [1]: from tgan.data import load_demo_data

In [2]: data, continuous_columns = load_demo_data('census')

In [3]: data.head(3).T[:10]
Out[3]:
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

In [4]: continuous_columns
Out[4]: [0, 5, 16, 17, 18, 29, 38]

```

### 2. Create a TGAN instance

The next step is to import TGAN and create an instance of the model.

To do so, we need to import the `tgan.model.TGANModel` class and call it.

```
In [5]: from tgan.model import TGANModel

In [6]: tgan = TGANModel(continuous_columns)
```

This will create a TGAN instance with the default parameters.

### 3. Fit the model

The third step is to pass the data that we have loaded previously to the `TGANModel.fit` method to
start the fitting.

```
In [7]: tgan.fit(data)
```

This process will not return anything, however, the progress of the fitting will be printed into
screen.

**NOTE** Depending on the performance of the system you are running, and the parameters selected
for the model, this step can take up to a few hours.

### 4. Sample new data

After the model has been fit, we are ready to generate new samples by calling the `TGANModel.sample`
method passing it the desired amount of samples:

```
In [8]: num_samples = 1000

In [9]: samples = tgan.sample(num_samples)

In [10]: samples.head(3)
Out[10]:

   age   Federal government   Local government   Never worked   Not in universe   Private  ...
0   59                    0                  0              0                 1         0  ...
1   37                    0                  0              0                 0         1  ...
2   18                    0                  0              1                 0         0  ...

```

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with
the same format as the input data and 1000 rows as we requested.

### 5. Save and Load a model

In the steps above we saw that the fitting process is slow, so we probably would like to avoid
having to fit every we want to generate samples. Instead we can fit a model once, save it, and
load it every time we want to sample new data.

If we have a fitted model, we can save it by calling the `TGANModel.save` method, that only takes
as argument the path to store the model into. Similarly, the `TGANModel.load` allows to load a
model stored on disk by passing as argument a path where the model is stored.

```
In [11]: model_path = 'models/mymodel'

In [12]: tgan.save(model_path)

In [13]: new_tgan = TGAN.load(model_path)

In [14]: new_samples = new_tgan.sample(num_samples)

In [15]: new_samples.head(3)
Out[15]:

   age   Federal government   Local government   Never worked   Not in universe   Private  ...
0   59                    0                  0              0                 1         0  ...
1   37                    0                  0              0                 0         1  ...
2   18                    0                  0              1                 0         0  ...
```

At this point we could use this model instance to generate more samples.

## Loading custom datasets

In the previous steps we used some demonstration data but we did not show how to load your own dataset.

In order to do so you can use `pandas.read_csv` by passing it the path to the CSV file that you want to load.

Additionally, you will need to create 0-indexed list of columns indices to be considered continuous.

For example, if we want to load a local CSV file, `path/to/my.csv`, that has as continuous columns
their first 4 columns, that is, indices `[0,1,2,3]`, we would do it like this:

```
In [16]: import pandas as pd

In [17]: data = pd.read_csv('data/census.csv')

In [18]: continuous_columns = [0,1,2,3]

```

## Model Parameters

If you want to change the default behavior of TGANModel, such as as different `batch_size` or
`num_epochs`, you can do so by passing different arguments when creating the instance. Have b

### Model general behavior

* continous_columns (`list[int]`, required): List of columns to be considered continuous.
* output (`str`, default=`output`): Path to store the model and its artifacts.
* gpu (`list[str]`, default=`[]`): Comma separated list of GPU(s) to use.

### Neural network definition and fitting

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

If we wanted to create an identical instance to the one created on step 2, but passing the arguments in a explicit way we will do something like this:

```
In [16]: tgan = TGANModel(
    ...:     continuous_columns,
    ...:     output='output',
    ...:     gpu=None,
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

## Citation

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
You can find the original paper [here](https://arxiv.org/pdf/1811.11264.pdf)