
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from  sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

def split_train_test(x, y, n_splits=5, test_size=0.2, seed=1024):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(x, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    return train_idxs, test_idxs



def load_wiki(view_data_dir, n_splits=5, idx_split=0, test_size=0.2, seed=1024):
    models_ls = ['v1','v2']  # Ego  'tags1k'
    view_train_x = []  ## 训练视图数据
    view_test_x = []  ## 测试视图数据
    for model in models_ls:
        view_train_x.append(np.load(os.path.join(view_data_dir, model + 'train.npy')))  ## 读取npy的数据
        view_test_x.append(np.load(os.path.join(view_data_dir, model + 'test.npy')))
    train_y = np.load(os.path.join(view_data_dir, 'train_y.npy'))
    test_y = np.load(os.path.join(view_data_dir, 'test_y.npy'))
    train_yy = []
    test_yy = []
    for y in train_y:
        train_yy.append(y - 1)
    for y in test_y:
        test_yy.append(y - 1)
    train_y = np.array(train_yy)
    test_y = np.array(test_yy)

    return view_train_x, train_y, view_test_x, test_y


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
        self.idx_split = idx_split
        view_data_dir = os.path.join('/data/tvgraz/')

        view_train_x, train_y, view_test_x, test_y = load_wiki(view_data_dir, n_splits=5, idx_split=self.idx_split, test_size=0.2, seed=1024)
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
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
