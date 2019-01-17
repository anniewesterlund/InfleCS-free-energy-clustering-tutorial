import numpy
from sklearn.model_selection import KFold


def split_train_validation(data, n_splits, shuffle=False):
    """
    Split the data into n_splits training and test sets
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    train_inds = []
    test_inds = []

    for train_ind, test_ind in kf.split(data):
        train_inds.append(train_ind)
        test_inds.append(test_ind)
    return train_inds, test_inds


def get_train_validation_set(data, train_ind, val_inds):
    """
    Get the train and test set given their sample/label indices
    """
    training_data = data[train_ind, :]
    validation_data = data[val_inds, :]

    return training_data, validation_data
