"""Functions to help during the preprocess of data."""
import json

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


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

    decorated.__doc__ == function.__doc__
    return decorated


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

    def reverse_transform(self, data, info):
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


class CategoricalTransformer:
    """One-hot encoder for Categorical transformer."""

    def transform(self, data):
        """Apply transform.

        Args:
            data(numpy.ndarray): Categorical array to transform.

        Return:
            tuple[numpy.ndarray, list, int]: Transformed values, list of unique values,
            and amount of uniques.

        """
        unique_values = np.unique(data).tolist()
        value_mapping = {value: index for index, value in enumerate(unique_values)}

        v = list(map(lambda x: value_mapping[x], data))
        features = np.asarray(v).reshape([-1, 1])

        return features, unique_values, len(unique_values)

    @check_inputs
    def reverse_transform(self, data, info):
        """Reverse the transform.

        Args:
            data(np.ndarray): Transformed data to restore as categorical.
            info(dict): Metadata for the given column.

        Returns:
            list: Values in the original space.

        """
        id2str = dict(enumerate(info['mapping']))
        return list(map(lambda x: id2str[x], data.flat))


def split_csv(csv_filename, csv_out1, csv_out2, ratio=0.8):
    """Split a csv file in two and save it.

    Args:
        csv_filename(str): Path for the original file.
        csv_out1(str): Destination for one of the splitted files.
        csv_out2(str): Destination for one of the splitted files.
        ratio(float): Size proportion to split the original file.

    Returns:
        None

    """
    df = pd.read_csv(csv_filename, header=-1)
    mask = np.random.rand(len(df)) < ratio
    df1 = df[mask]
    df2 = df[~mask]
    df1.to_csv(csv_out1, header=False, index=False)
    df2.to_csv(csv_out2, header=False, index=False)


def csv_to_npz(csv_filename, npz_filename, continuous_cols):
    """Read data from a csv file and convert it to the training npz for TGAN.

    Args:
        csv_filename(str): Path to origin csv file.
        npz_filename(str): Path to store the destination npz file.
        continuous_cols(list[str or int]): List of labels for columns with continous values.

    Returns:
        None

    """
    df = pd.read_csv(csv_filename, header=-1)
    num_cols = len(list(df))

    data = {}
    details = []
    continous_transformer = MultiModalNumberTransformer()
    categorical_transformer = CategoricalTransformer()

    for i in range(num_cols):
        if i in continuous_cols:
            column_data = df[i].values.reshape([-1, 1])
            features, probs, means, stds = continous_transformer.transform(column_data)
            details.append({
                "type": "value",
                "means": means,
                "stds": stds,
                "n": 5
            })
            data['f%02d' % i] = np.concatenate((features, probs), axis=1)

        else:
            column_data = df[i].astype(str).values
            features, mapping, n = categorical_transformer.transform(column_data)
            data['f%02d' % i] = features
            details.append({
                "type": "category",
                "mapping": mapping,
                "n": n
            })

    info = {
        "num_features": num_cols,
        "details": details
    }

    np.savez(npz_filename, info=json.dumps(info), **data)


def npz_to_csv(npfilename, csvfilename):
    """Convert a npz file into a csv and return its contents.

    Args:
        npfilename(str): Path to origin npz file.
        csvfilename(str): Path to destination csv file.

    Returns:
        None

    """
    data = np.load(npfilename)
    metadata = json.loads(str(data['info']))
    check_metadata(metadata)

    table = []
    continous_transformer = MultiModalNumberTransformer()
    categorical_transformer = CategoricalTransformer()

    for i in range(metadata['num_features']):
        column_data = data['f%02d' % i]
        column_metadata = metadata['details'][i]

        if column_metadata['type'] == 'value':
            column = continous_transformer.reverse_transform(column_data, column_metadata)

        if column_metadata['type'] == 'category':
            column = categorical_transformer.reverse_transform(column_data, column_metadata)

        table.append(column)

    df = pd.DataFrame(dict(enumerate(table)))
    df.to_csv(csvfilename, index=False, header=False)
