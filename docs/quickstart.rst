Quickstart
==========

In this section we will show the most basic usage of **TGAN** in order to generate samples from a
given dataset.

**NOTE**: All the examples of this tutorial are run in an `IPython Shell`, which you can install
by running the following commands:

.. code-block:: bash

    pip install ipython
    ipython


1. Load the data
----------------

The first step is to load the data wich we will use to fit TGAN. In order to do so, we will first
import the function ``tgan.data.load_demo_data`` and call it with the name the dataset that we
want to load.

In this case, we will load the ``census`` dataset, which we will use during the subsequent steps,
and obtain two objects:

1. ``data`` will contain a ``pandas.DataFrame`` with the table of data from the ``census`` dataset
   ready to be used to fit the model.

2. ``continuous_columns`` will contain a ``list`` with the indices of continuous columns.

.. code-block:: python

    In [1]: from tgan.data import load_demo_data

    In [2]: data, continuous_columns = load_demo_data('census')

    In [3]: data.head(2).T[:10]
    Out[3]:
                                  0                                     1
    0                            73                                    58
    1               Not in universe        Self-employed-not incorporated
    2                             0                                     4
    3                             0                                    34
    4          High school graduate            Some college but no degree
    5                             0                                     0
    6               Not in universe                       Not in universe
    7                       Widowed                              Divorced
    8   Not in universe or children                          Construction
    9               Not in universe   Precision production craft & repair

    In [4]: continuous_columns
    Out[4]: [0, 5, 16, 17, 18, 29, 38]


2. Create a TGAN instance
-------------------------

The next step is to import TGAN and create an instance of the model.

To do so, we need to import the `tgan.model.TGANModel` class and call it.

.. code-block:: python

    In [5]: from tgan.model import TGANModel

    In [6]: tgan = TGANModel(continuous_columns)


This will create a TGAN instance with the default parameters.

3. Fit the model
----------------

The third step is to pass the data that we have loaded previously to the `TGANModel.fit` method to
start the fitting.

.. code-block:: python

    In [7]: tgan.fit(data)

This process will not return anything, however, the progress of the fitting will be printed into
screen.

**NOTE** Depending on the performance of the system you are running, and the parameters selected
for the model, this step can take up to a few hours.

4. Sample new data
------------------

After the model has been fit, we are ready to generate new samples by calling the
`TGANModel.sample` method passing it the desired amount of samples:

.. code-block:: python

    >>> num_samples = 1000
    >>> samples = tgan.sample(num_samples)
    >>> samples.head(3)

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with
the same format as the input data and 1000 rows as we requested.

5. Save and Load a model
------------------------

In the steps above we saw that the fitting process is slow, so we probably would like to avoid
having to fit every we want to generate samples. Instead we can fit a model once, save it, and
load it every time we want to sample new data.

If we have a fitted model, we can save it by calling the `TGANModel.save` method, that only takes
as argument the path to store the model into. Similarly, the `TGANModel.load` allows to load a
model stored on disk by passing as argument a path where the model is stored.

.. code-block:: python

    >>> model_path = 'models/mymodel'
    >>> tgan.save(model_path)
    >>> new_tgan = TGAN.load(model_path)
    >>> new_samples = new_tgan.sample(num_samples)
    >>> new_samples.head(3)


At this point we could use this model instance to generate more samples.


Data Format
-----------

Input
+++++

In order to be able to sample new synthetic data, **TGAN** needs to first be *fitted* to existing
data.

The input data for this *fitting* process has to be a single table that:

* Has no missing values.
* Has columns of types ``int``, ``float``, ``str`` or ``bool``.
* Each column contains data of only one type.

The following is a simple example of a table with 4 columns, ``str_column``, ``float_column``,
``int_column``, ``bool_column``, each one being an example of one of the supported value types.


+------------+--------------+------------+-------------+
| str_column | float_column | int_column | bool_column |
+------------+--------------+------------+-------------+
| 'green'    | 0.15         | 10         | True        |
+------------+--------------+------------+-------------+
| 'blue'     | 7.25         | 23         | False       |
+------------+--------------+------------+-------------+
| 'red'      | 10.00        | 1          | False       |
+------------+--------------+------------+-------------+
| 'yellow'   | 5.50         | 17         | True        |
+------------+--------------+------------+-------------+

**NOTE**: It's important to have properly identifed which of the columns are numerical, which means
that they represent a magnitude, and which ones are categorical, as during the preprocessing of
the data, numerical and categorical columns will be processed differently.

Output
++++++

The output of **TGAN** is a table of sampled data with the same columns as the input table and as
many rows as requested.

Demo Datasets
-------------

**TGAN** includes a few datasets to use for development or demonstration purposes. These datasets
come from the `UCI Machine Learning repository`_, but have been
preprocessed to be ready to use with **TGAN**, following the requirements specified in the `Input`
section.

These datasets can be browsed and directly downloaded from the `tgan-demo AWS S3 Bucket`_.

Census dataset
++++++++++++++

This dataset contains a single table, with information from the census, labeled with information
of wheter or not the income of is greater than 50.000 $/year. It's a single csv file, containing
199522 rows and 41 columns. From these 41 columns, only 7 are identified as continuous.
In **TGAN** this dataset is called :attr:`census`.

Cover type
++++++++++

This dataset contains a single table with cartographic information labeled with the different
forrest cover types. It's a single csv file, containing 465588 rows and 55 columns. From these
55 columns, 10 are identified as continuous. In **TGAN** this dataset is called :attr:`covertype`.


.. _UCI Machine Learning repository: http://archive.ics.uci.edu/ml
.. _tgan-demo AWS S3 Bucket: https://s3.amazonaws.com/hdi-demos/tgan-demo/
.. _IPython Shell: https://ipython.org/
