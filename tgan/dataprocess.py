"""Functions to help during the preprocess of data."""
import json

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


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


def value_clustering(data, n):
    """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.

    Args:
        data(numpy.ndarray): Values to cluster in array of shape (n,1).
        n(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.

    Returns:
        tuple[numpy.array, numpy.array, list, list]: Tuple containg the features, probabilities,
        averages and stds of the given data.

    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html

    """
    if not (isinstance(data, np.ndarray) and data.shape[1] == 1):
        raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

    model = GaussianMixture(n)
    model.fit(data)

    means = model.means_.reshape((1, n))
    stds = np.sqrt(model.covariances_).reshape((1, n))

    features = (data - means) / (2 * stds)
    probs = model.predict_proba(data)
    argmax = np.argmax(probs, axis=1)
    idx = np.arange(len(features))
    features = features[idx, argmax].reshape([-1, 1])

    features = np.clip(features, -0.99, 0.99)
    return features, probs, list(means.flat), list(stds.flat)


def csv_to_npz(csv_filename, npz_filename, continuous_cols):
    """Read data from a csv file and convert it to the training npz for TGAN.

    Args:
        csv_filename(str): Path to origin csv file.
        npz_filename(str): Path to store the destination npz file.
        continuous_cols(list[str or int]): List of labels for columns with continous values.

    Returns:
        tuple[dict, dict]: The first element contains the actual arrays stored in the file,
        the second is the metadata.

        Please note that this function returns this after the actual saving.

    """
    df = pd.read_csv(csv_filename, header=-1)
    num_cols = len(list(df))

    data = {}
    details = []

    for i in range(num_cols):

        if i in continuous_cols:
            features, probs, means, stds = value_clustering(df[i].values.reshape([-1, 1]), 5)
            details.append({
                "type": "value",
                "means": means,
                "stds": stds,
                "n": 5
            })
            data['f%02d' % i] = np.concatenate((features, probs), axis=1)

        else:
            df[i] = df[i].astype(str)
            vset = list(df[i].unique())
            vdict = {}
            for idx, v in enumerate(vset):
                vdict[v] = idx

            v = df[i].values
            v = list(map(lambda x: vdict[x], v))
            v = np.asarray(v).reshape([-1, 1])

            data['f%02d' % i] = v

            details.append({
                "type": "category",
                "mapping": vset,
                "n": len(vset)
            })

    info = {
        "num_features": num_cols,
        "details": details
    }

    np.savez(npz_filename, info=json.dumps(info), **data)
    return data, info


def _rev_feature(data, info):
    """Reverse the clustering on values.

    Args:
        data(numpy.ndarray): Data to transform.
        info(dict): Metadata.

    Returns:
        Iterable: If info['type'] == value, returns a :class:`numpy.ndarray`,
        if info['type'] == 'category' returns a :class:`list`.

    """
    if info['type'] == 'value':
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(info['means'])
        std = np.asarray(info['stds'])

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        return features * 2 * select_std + select_mean

    elif info['type'] == 'category':
        if not(len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        id2str = dict(enumerate(info['mapping']))
        return list(map(lambda x: id2str[x], data.flat))

    else:
        raise ValueError(
            "info['type'] must be either 'category' or 'value'."
            "Instead it was '{}'.".format(info['type'])
        )


def npz_to_csv(npfilename, csvfilename):
    """Convert a npz file into a csv and return its contents.

    Args:
        npfilename(str): Path to origin npz file.
        csvfilename(str): Path to destination csv file.

    Returns:
        pandas.DataFrame: Formatted contents of :attr:`npfilename`

    """
    data = np.load(npfilename)
    info = json.loads(str(data['info']))

    table = []
    for i in range(info['num_features']):
        feature = data['f%02d' % i]
        col = _rev_feature(feature, info['details'][i])
        table.append(col)

    df = pd.DataFrame(dict(enumerate(table)))
    df.to_csv(csvfilename, index=False, header=False)
    return df


if __name__ == "__main__":
    csv_to_npz("census-test.csv", "census.npz", [0, 5, 16, 17, 18, 29, 38])
    npz_to_csv("census.npz", "reconstruct.csv")
