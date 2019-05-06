"""Data related functionalities.

This modules contains the tools to preprare the data, from the raw csv files, to the DataFlow
objects will be used to fit our models.
"""
import os
import urllib

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from tensorpack import DataFlow, RNGDataFlow

DEMO_DATASETS = {
    'census': (
        'http://hdi-project-tgan.s3.amazonaws.com/census-train.csv',
        'data/census.csv',
        [0, 5, 16, 17, 18, 29, 38]
    ),
    'covertype': (
        'http://hdi-project-tgan.s3.amazonaws.com/covertype-train.csv',
        'data/covertype.csv',
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
}


def check_metadata(metadata):
    """Check that the given metadata has correct types for all its members.

    Args:
        metadata(dict): Description of the inputs.

    Returns:
        None

    Raises:
        AssertionError: If any of the details is not valid.

    """
    message = 'The given metadata contains unsupported types.'
    assert all([item['type'] in ['category', 'value'] for item in metadata['details']]), message


def check_inputs(function):
    """Validate inputs for functions whose first argument is a numpy.ndarray with shape (n,1).

    Args:
        function(callable): Method to validate.

    Returns:
        callable: Will check the inputs before calling :attr:`function`.

    Raises:
        ValueError: If first argument is not a valid :class:`numpy.array` of shape (n, 1).

    """
    def decorated(self, data, *args, **kwargs):
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated


class TGANDataFlow(RNGDataFlow):
    """Subclass of :class:`tensorpack.RNGDataFlow` prepared to work with :class:`numpy.ndarray`.

    Attributes:
        shuffle(bool): Wheter or not to shuffle the data.
        metadata(dict): Metadata for the given :attr:`data`.
        num_features(int): Number of features in given data.
        data(list): Prepared data from :attr:`filename`.
        distribution(list): DepecrationWarning?

    """

    def __init__(self, data, metadata, shuffle=True):
        """Initialize object.

        Args:
            filename(str): Path to the json file containing the metadata.
            shuffle(bool): Wheter or not to shuffle the data.

        Raises:
            ValueError: If any column_info['type'] is not supported

        """
        self.shuffle = shuffle
        if self.shuffle:
            self.reset_state()

        self.metadata = metadata
        self.num_features = self.metadata['num_features']

        self.data = []
        self.distribution = []
        for column_id, column_info in enumerate(self.metadata['details']):
            if column_info['type'] == 'value':
                col_data = data['f%02d' % column_id]
                value = col_data[:, :1]
                cluster = col_data[:, 1:]
                self.data.append(value)
                self.data.append(cluster)

            elif column_info['type'] == 'category':
                col_data = np.asarray(data['f%02d' % column_id], dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "column_info['type'] must be either 'category' or 'value'."
                    "Instead it was '{}'.".format(column_info['type'])
                )

        self.data = list(zip(*self.data))

    def size(self):
        """Return the number of rows in data.

        Returns:
            int: Number of rows in :attr:`data`.

        """
        return len(self.data)

    def get_data(self):
        """Yield the rows from :attr:`data`.

        Yields:
            tuple: Row of data.

        """
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

    def __iter__(self):
        """Iterate over self.data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.size()


class RandomZData(DataFlow):
    """Random dataflow.

    Args:
        shape(tuple): Shape of the array to return on :meth:`get_data`

    """

    def __init__(self, shape):
        """Initialize object."""
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        """Yield random normal vectors of shape :attr:`shape`."""
        while True:
            yield [np.random.normal(0, 1, size=self.shape)]

    def __iter__(self):
        """Return data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.shape[0]


class MultiModalNumberTransformer:
    r"""Reversible transform for multimodal data.

    To effectively sample values from a multimodal distribution, we cluster values of a
    numerical variable using a `skelarn.mixture.GaussianMixture`_ model (GMM).

    * We train a GMM with :attr:`n` components for each numerical variable :math:`C_i`.
      GMM models a distribution with a weighted sum of :attr:`n` Gaussian distributions.
      The means and standard deviations of the :attr:`n` Gaussian distributions are
      :math:`{\eta}^{(1)}_{i}, ..., {\eta}^{(n)}_{i}` and
      :math:`{\sigma}^{(1)}_{i}, ...,{\sigma}^{(n)}_{i}`.

    * We compute the probability of :math:`c_{i,j}` coming from each of the :attr:`n` Gaussian
      distributions as a vector :math:`{u}^{(1)}_{i,j}, ..., {u}^{(n)}_{i,j}`. u_{i,j} is a
      normalized probability distribution over :attr:`n` Gaussian distributions.

    * We normalize :math:`c_{i,j}` as :math:`v_{i,j} = (c_{i,j}−{\eta}^{(k)}_{i})/2{\sigma}^
      {(k)}_{i}`, where :math:`k = arg max_k {u}^{(k)}_{i,j}`. We then clip :math:`v_{i,j}` to
      [−0.99, 0.99].

    Then we use :math:`u_i` and :math:`v_i` to represent :math:`c_i`. For simplicity,
    we cluster all the numerical features, i.e. both uni-modal and multi-modal features are
    clustered to :attr:`n = 5` Gaussian distributions.

    The simplification is fair because GMM automatically weighs :attr:`n` components.
    For example, if a variable has only one mode and fits some Gaussian distribution, then GMM
    will assign a very low probability to :attr:`n − 1` components and only 1 remaining
    component actually works, which is equivalent to not clustering this feature.

    Args:
        num_modes(int): Number of modes on given data.

    Attributes:
        num_modes(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.

    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html

    """

    def __init__(self, num_modes=5):
        """Initialize instance."""
        self.num_modes = num_modes

    @check_inputs
    def transform(self, data):
        """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.

        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.

        .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
            sklearn.mixture.GaussianMixture.html

        """
        model = GaussianMixture(self.num_modes)
        model.fit(data)

        means = model.means_.reshape((1, self.num_modes))
        stds = np.sqrt(model.covariances_).reshape((1, self.num_modes))

        features = (data - means) / (2 * stds)
        probs = model.predict_proba(data)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape([-1, 1])

        features = np.clip(features, -0.99, 0.99)

        return features, probs, list(means.flat), list(stds.flat)

    @staticmethod
    def inverse_transform(data, info):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.

        Returns:
           numpy.ndarray: Values in the original space.

        """
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(info['means'])
        std = np.asarray(info['stds'])

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        return features * 2 * select_std + select_mean


class Preprocessor:
    """Transform back and forth human-readable data into TGAN numerical features.

    Args:
        continous_columns(list): List of columns to be considered continuous
        metadata(dict): Metadata to initialize the object.

    Attributes:
        continous_columns(list): Same as constructor argument.
        metadata(dict): Information about the transformations applied to the data and its format.
        continous_transformer(MultiModalNumberTransformer):
            Transformer for columns in :attr:`continuous_columns`
        categorical_transformer(CategoricalTransformer):
            Transformer for categorical columns.
        columns(list): List of columns labels.

    """

    def __init__(self, continuous_columns=None, metadata=None):
        """Initialize object, set arguments as attributes, initialize transformers."""
        if continuous_columns is None:
            continuous_columns = []

        self.continuous_columns = continuous_columns
        self.metadata = metadata
        self.continous_transformer = MultiModalNumberTransformer()
        self.categorical_transformer = LabelEncoder()
        self.columns = None

    def fit_transform(self, data, fitting=True):
        """Transform human-readable data into TGAN numerical features.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        num_cols = data.shape[1]
        self.columns = data.columns
        data.columns = list(range(num_cols))

        transformed_data = {}
        details = []

        for i in data.columns:
            if i in self.continuous_columns:
                column_data = data[i].values.reshape([-1, 1])
                features, probs, means, stds = self.continous_transformer.transform(column_data)
                transformed_data['f%02d' % i] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details.append({
                        "type": "value",
                        "means": means,
                        "stds": stds,
                        "n": 5
                    })

            else:
                column_data = data[i].astype(str).values
                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data['f%02d' % i] = features.reshape([-1, 1])

                if fitting:
                    mapping = self.categorical_transformer.classes_
                    details.append({
                        "type": "category",
                        "mapping": mapping,
                        "n": mapping.shape[0],
                    })

        if fitting:
            metadata = {
                "num_features": num_cols,
                "details": details
            }
            check_metadata(metadata)
            self.metadata = metadata

        return transformed_data

    def transform(self, data):
        """Transform the given dataframe without generating new metadata.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        return self.fit_transform(data, fitting=False)

    def fit(self, data):
        """Initialize the internal state of the object using :attr:`data`.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        self.fit_transform(data)

    def reverse_transform(self, data):
        """Transform TGAN numerical features back into human-readable data.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        table = []

        for i in range(self.metadata['num_features']):
            column_data = data['f%02d' % i]
            column_metadata = self.metadata['details'][i]

            if column_metadata['type'] == 'value':
                column = self.continous_transformer.inverse_transform(column_data, column_metadata)

            if column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = column_metadata['mapping']
                column = self.categorical_transformer.inverse_transform(
                    column_data.ravel().astype(np.int32))

            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        result.columns = self.columns
        return result


def load_demo_data(name, header=None):
    """Fetch, load and prepare a dataset.

    If name is one of the demo datasets


    Args:
        name(str): Name or path of the dataset.
        header(): Header parameter when executing :attr:`pandas.read_csv`

    """
    params = DEMO_DATASETS.get(name)
    if params:
        url, file_path, continuous_columns = params
        if not os.path.isfile(file_path):
            base_path = os.path.dirname(file_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            urllib.request.urlretrieve(url, file_path)

    else:
        message = (
            '{} is not a valid dataset name. '
            'Supported values are: {}.'.format(name, list(DEMO_DATASETS.keys()))
        )
        raise ValueError(message)

    return pd.read_csv(file_path, header=header), continuous_columns
