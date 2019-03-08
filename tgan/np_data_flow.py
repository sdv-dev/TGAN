"""This module contains a wrapper of RNGDataFlow."""

import json

import numpy as np
from tensorpack import BatchData, RNGDataFlow


class NpDataFlow(RNGDataFlow):
    """Subclass of :class:`tensorpack.RNGDataFlow` prepared to work with :class:`numpy.ndarray`.

    Attributes:
        shuffle(bool): Wheter or not to shuffle the data.
        info(dict): Metadata for the given :attr:`data`.
        num_features(int): Number of features in given data.
        data(list): Prepared data from :attr:`filename`.
        distribution(list): DepecrationWarning?

    """

    def __init__(self, filename, shuffle=True):
        """Initialize object.

        Args:
            filename(str): Path to the json file containing the metadata.
            shuffle(bool): Wheter or not to shuffle the data.

        Raises:
            ValueError: If any col_info['type'] is not supported

        """
        self.shuffle = shuffle
        data = np.load(filename)

        self.info = json.loads(str(data['info']))
        self.num_features = self.info['num_features']

        self.data = []
        self.distribution = []
        for col_id, col_info in enumerate(self.info['details']):
            if col_info['type'] == 'value':
                col_data = data['f%02d' % col_id]
                value = col_data[:, :1]
                cluster = col_data[:, 1:]
                self.data.append(value)
                self.data.append(cluster)

            elif col_info['type'] == 'category':
                col_data = np.asarray(data['f%02d' % col_id], dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "col_info['type'] must be either 'category' or 'value'."
                    "Instead it was '{}'.".format(col_info['type'])
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


if __name__ == '__main__':
    ds = NpDataFlow('census-2c.npz', shuffle=False)
    print(ds.distribution)
    ds = BatchData(ds, 3)

    for i in ds.get_data():
        print(i[0], i[2])
        print(len(i))
        break
