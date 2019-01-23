import numpy as np
import pandas as pd
import json
from sklearn.mixture import GaussianMixture

def split_csv(csv_filename, csv_out1, csv_out2, ratio=0.8):
    df = pd.read_csv(csv_filename, header=-1)
    mask = np.random.rand(len(df)) < ratio
    df1 = df[mask]
    df2 = df[~mask]
    df1.to_csv(csv_out1, header=False, index=False)
    df2.to_csv(csv_out2, header=False, index=False)

def value_clustering(data, n):
    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 1
    model = GaussianMixture(n)
    model.fit(data)

    weights = model.weights_
    means = model.means_.reshape((1, n))
    stds = np.sqrt(model.covariances_).reshape((1, n))

    features = (data - means) / (2 * stds)
    probs = model.predict_proba(data)
    argmax = np.argmax(probs, axis=1)
    idx = np.arange(len(features))
    features = features[idx, argmax].reshape([-1, 1])

    features = np.clip(features, -.99, .99)
    return features, probs, list(means.flat), list(stds.flat)

def csv_to_npz(csv_filename, npz_filename, continuous_cols):
    """csv_to_npz reads data from a csv file and convert it to
    the training npz for TGAN.
    Args:
        csv_filename:
        npz_filename:
        continuous_cols:
    return:
        data:
        info:
    """

    df = pd.read_csv(csv_filename, header=-1)
    num_cols = len(list(df))

    data = {}
    info = []

    for i in range(num_cols):

        if i in continuous_cols:
            features, probs, means, stds = value_clustering(df[i].values.reshape([-1, 1]), 5)
            info.append({
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

            info.append({
                "type": "category",
                "mapping": vset,
                "n": len(vset)
            })

    info = {
        "num_features": num_cols,
        "details": info
    }

    np.savez(npz_filename, info=json.dumps(info), **data)
    return data, info


def _rev_feature(data, info):
    if info['type'] == 'value':
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(info['means'])
        std = np.asarray(info['stds'])

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        values = features * 2 * select_std + select_mean
        return values

    elif info['type'] == 'category':
        id2str = dict(enumerate(info['mapping']))
        assert len(data.shape) == 2
        assert data.shape[1] == 1
        categories = list(map(lambda x: id2str[x], data.flat))
        return categories
    else:
        assert 0

def npz_to_csv(npfilename, csvfilename):
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
