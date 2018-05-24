from tensorpack import *
import numpy as np
import json

class NpDataFlow(RNGDataFlow):
    def __init__(self, filename, shuffle=True):
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
                assert 0

        self.data = list(zip(*self.data))

    def size(self):
        return len(self.data)

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

if __name__ == "__main__":
    ds = NpDataFlow('census-2c.npz', shuffle=False)
    print(ds.distribution)
    ds = BatchData(ds, 3)


    for i in ds.get_data():
        print(i[0], i[2])
        print(len(i))
        break
