import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics

def _proc_data(df, continuous_cols):
    features = []
    num_cols = len(list(df))

    for i in range(num_cols - 1):
        if i in continuous_cols:
            features.append(df[i].values.reshape([-1, 1]))
        else:
            features.append(pd.get_dummies(df[i]).values)
    features = np.concatenate(features, axis=1)
    labels = df[num_cols - 1].values
    return features, labels

def evaluate_classification(
    train_csv_filename,
    test_csv_filename,
    continuous_cols,
    classifier=DecisionTreeClassifier(max_depth=20),
    metric=sklearn.metrics.accuracy_score):

    train_set = pd.read_csv(train_csv_filename, header=-1)
    test_set = pd.read_csv(test_csv_filename, header=-1)

    n_train = len(train_set)
    n_test = len(test_set)
    dataset = pd.concat([train_set, test_set])

    features, labels = _proc_data(dataset, continuous_cols)

    train_set = features[:n_train], labels[:n_train]
    test_set = features[n_train:], labels[n_train:]

    classifier.fit(train_set[0], train_set[1])

    pred = classifier.predict(test_set[0])
    return metric(test_set[1], pred)

if __name__ == "__main__":
    print(evaluate_classification('reconstruct.csv', 'census-test.csv', [0, 5, 16, 17, 18, 29, 38]))
