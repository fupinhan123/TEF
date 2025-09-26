
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from torch.utils.data import Dataset


def split_train_test(x, y, n_splits=5, test_size=0.2, seed=1024):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(x, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    return train_idxs, test_idxs

def load_nus_wide(view_data_dir, n_splits=5, idx_split=1, test_size=0.2, seed=1024):
    print('********************** idx_split:', idx_split)
    view_names = ['Normalized_CH', 'Normalized_CM55', 'Normalized_CORR', 'Normalized_EDH', 'Normalized_WT', 'BoW_int', 'tags1k']
    x = np.load(os.path.join(view_data_dir, view_names[-1] + '.npy'))
    y = np.load(os.path.join(view_data_dir, 'y.npy'))

    train_idxs, test_idxs = split_train_test(x=x, y=y, n_splits=n_splits,
                                            test_size=test_size, seed=seed)
    view_train_x, view_test_x = [], []
    for view_name in view_names:
        x = np.load(os.path.join(view_data_dir, view_name+'.npy'))
        view_train_x.append(x[train_idxs[idx_split]])
        view_test_x.append(x[test_idxs[idx_split]])
    train_y = y[train_idxs[idx_split]]
    test_y = y[test_idxs[idx_split]]

    ##在这里多读一个

    x_train = np.load(os.path.join('/export/fupinhan/datasetENAS/nus_wide_view_result/data/',
                                   'output_data_train.npy'))
    x_train_y = np.load(os.path.join('/export/fupinhan/datasetENAS/nus_wide_view_result/data/',
                                     'output_data_train_y.npy'))

    x_test = np.load(os.path.join('/export/fupinhan/datasetENAS/nus_wide_view_result/data/',
                                  'output_data_test.npy'))
    x_test_y = np.load(os.path.join('/export/fupinhan/datasetENAS/nus_wide_view_result/data/',
                                    'output_data_test_y.npy'))

    view_train_x.append(x_train)
    view_test_x.append(x_test)

    view_train_x.append(x_train)
    view_test_x.append(x_test)

    return view_train_x, train_y, view_test_x, test_y

def split_train_val_test(x, y, train_size=0.7, val_size=0.1, test_size=0.2, seed=1024):
    train_idxs, val_idxs, test_idxs = [], [], []
    train_val_idxs = []

    # Step 1: Split into train+val and test sets
    sss_temp = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)
    for train_val_idx, test_idx in sss_temp.split(x, y):
        train_val_idxs.append(train_val_idx)
        test_idxs.append(test_idx)

    # Step 2: Split the train+val set into train and val sets
    for i in range(5):
        train_val_idx = train_val_idxs[i]
        sss_final = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (train_size + val_size), random_state=seed)
        train_idx, val_idx = next(sss_final.split(x[train_val_idx], y[train_val_idx]))

        # Adjust indices to match the original dataset
        train_idxs.append(train_val_idx[train_idx])
        val_idxs.append(train_val_idx[val_idx])

    return train_idxs, val_idxs, test_idxs


def merge_train_val(view_train_x, train_y, view_val_x, val_y):
    merged_view_x = [np.concatenate((train_x, val_x), axis=0) for train_x, val_x in zip(view_train_x, view_val_x)]
    merged_y = np.concatenate((train_y, val_y), axis=0)
    return merged_view_x, merged_y

def load_nus_wide_new(view_data_dir, train_size=0.7, val_size=0.1, test_size=0.2, seed=1024, idx_split=0):
    print('Loading data...')
    view_names = ['Normalized_CH', 'Normalized_CM55', 'Normalized_CORR', 'Normalized_EDH', 'Normalized_WT', 'BoW_int',
                  'tags1k']

    # Load the data
    x = np.load(os.path.join(view_data_dir, view_names[-1] + '.npy'))
    y = np.load(os.path.join(view_data_dir, 'y.npy'))

    # Get indices for splitting
    train_idxs, val_idxs, test_idxs = split_train_val_test(x=x, y=y, train_size=train_size, val_size=val_size,
                                                           test_size=test_size, seed=seed)

    # Prepare the data for each view
    view_train_x, view_val_x, view_test_x = [], [], []
    for view_name in view_names:
        x_view = np.load(os.path.join(view_data_dir, view_name + '.npy'))
        view_train_x.append(x_view[train_idxs[idx_split]])
        view_val_x.append(x_view[val_idxs[idx_split]])
        view_test_x.append(x_view[test_idxs[idx_split]])

    train_y = y[train_idxs[idx_split]]
    val_y = y[val_idxs[idx_split]]
    test_y = y[test_idxs[idx_split]]

    print(len(train_y), len(val_y), len(test_y))

    return view_train_x, train_y, view_val_x, val_y, view_test_x, test_y



class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, train=True, idx_split=0):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        self.idx_split = idx_split
        view_data_dir = os.path.join('/export/nus_wide/view')

        #view_train_x, train_y, view_test_x, test_y = load_nus_wide(view_data_dir, n_splits=5, idx_split=self.idx_split, test_size=0.2, seed=1024)
        view_train_x, train_y,view_test_x,test_y = load_nus_wide(view_data_dir,n_splits=5,idx_split=self.idx_split, test_size=0.2, seed=1024)

        view_number = len(view_train_x)
        self.X = dict()
        if train:
            for v_num in range(view_number):
                self.X[v_num] = normalize(view_train_x[v_num])
                print(self.X[v_num].shape)
            y = train_y
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(view_test_x[v_num])
            y = test_y

        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    return x
    # if min == 0:
    #     scaler = MinMaxScaler([0, 1])
    # else:  # min=-1
    #     scaler = MinMaxScaler((-1, 1))
    # norm_x = scaler.fit_transform(x)
    # return norm_x
