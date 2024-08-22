import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import pywt
import matplotlib.pyplot as plt
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class UEAloader(Dataset):
#     """
#     Dataset class for datasets included in:
#         Time Series Classification Archive (www.timeseriesclassification.com)
#     Argument:
#         limit_size: float in (0, 1) for debug
#     Attributes:
#         all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
#         labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """
#
#     def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
#         self.args = args
#         self.root_path = root_path
#         self.flag = flag
#         self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
#         self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
#
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#
#         # use all features
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df
#         print(self.feature_df)
#
#         # pre_process
#         normalizer = Normalizer()
#         self.feature_df = normalizer.normalize(self.feature_df)
#         print(len(self.all_IDs))
#
#     def load_all(self, root_path, file_list=None, flag=None):
#         """
#         Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
#         Args:
#             root_path: directory containing all individual .csv files
#             file_list: optionally, provide a list of file paths within `root_path` to consider.
#                 Otherwise, entire `root_path` contents will be used.
#         Returns:
#             all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
#             labels_df: dataframe containing label(s) for each sample
#         """
#         # Select paths for training and evaluation
#         if file_list is None:
#             data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
#         else:
#             data_paths = [os.path.join(root_path, p) for p in file_list]
#         if len(data_paths) == 0:
#             raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
#         if flag is not None:
#             data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
#         input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
#         if len(input_paths) == 0:
#             pattern='*.ts'
#             raise Exception("No .ts files found using pattern: '{}'".format(pattern))
#
#         all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset
#
#         return all_df, labels_df
#
#     def load_single(self, filepath):
#         df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
#                                                              replace_missing_vals_with='NaN')
#         labels = pd.Series(labels, dtype="category")
#         self.class_names = labels.cat.categories
#         labels_df = pd.DataFrame(labels.cat.codes,
#                                  dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
#
#         lengths = df.applymap(
#             lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
#
#         horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
#
#         if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
#             df = df.applymap(subsample)
#
#         lengths = df.applymap(lambda x: len(x)).values
#         vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
#         if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
#             self.max_seq_len = int(np.max(lengths[:, 0]))
#         else:
#             self.max_seq_len = lengths[0, 0]
#
#         # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
#         # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
#         # sample index (i.e. the same scheme as all datasets in this project)
#
#         df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
#             pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)
#
#         # Replace NaN values
#         grp = df.groupby(by=df.index)
#         df = grp.transform(interpolate_missing)
#
#         return df, labels_df
#
#     def instance_norm(self, case):
#         if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case
#
#     def __getitem__(self, ind):
#         batch_x = self.feature_df.loc[self.all_IDs[ind]].values
#         labels = self.labels_df.loc[self.all_IDs[ind]].values
#         if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
#             num_samples = len(self.all_IDs)
#             num_columns = self.feature_df.shape[1]
#             seq_len = int(self.feature_df.shape[0] / num_samples)
#             batch_x = batch_x.reshape((1, seq_len, num_columns))
#             batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)
#
#             batch_x = batch_x.reshape((1 * seq_len, num_columns))
#
#         return self.instance_norm(torch.from_numpy(batch_x)), \
#                torch.from_numpy(labels)
#
#     def __len__(self):
#         return len(self.all_IDs)


##feng ge hou de shuju
# class UEAloader(Dataset):
#     """
#     Dataset class for datasets included in:
#         Time Series Classification Archive (www.timeseriesclassification.com)
#     Argument:
#         limit_size: float in (0, 1) for debug
#     Attributes:
#         all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
#         labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """
#
#     def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
#         self.args = args
#         self.root_path = root_path
#         self.all_df, self.labels_df, self.class_names = self.load_all(root_path, file_list=file_list, flag=flag)
#
#         self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
#
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#
#         # use all features
#         self.max_seq_len=3000
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df
#         print('self.feature_df', len(self.feature_df))
#         print(self.feature_df)
#         print('self.labels_df', len(self.labels_df))
#         print(self.labels_df)
#         # pre_process
#         normalizer = Normalizer()
#         self.feature_df = normalizer.normalize(self.feature_df)
#
#         # 打印 class_names
#         print(f'class_names:',self.class_names)
#
#     def load_all(self, root_path, file_list=None, flag=None):
#         """
#         Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
#         Args:
#             root_path: directory containing all individual .csv files
#             file_list: optionally, provide a list of file paths within `root_path` to consider.
#                 Otherwise, entire `root_path` contents will be used.
#         Returns:
#             all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
#             labels_df: dataframe containing label(s) for each sample
#             class_names: list of class names corresponding to labels
#         """
#         all_dfs = []
#         labels = []
#         index_counter = 0
#         # Determine the subdirectory based on the flag
#         if flag not in ['TRAIN', 'TEST']:
#             raise ValueError("Flag must be 'TRAIN' or 'TEST'")
#
#         data_path = os.path.join(root_path, flag)
#         # Traverse through all subdirectories
#         for root, dirs, files in os.walk(data_path):
#             for file in files:
#                 if file.endswith('.csv'):
#                     filepath = os.path.join(root, file)
#                     df = pd.read_csv(filepath, header=0)  # Read CSV with header
#                     # if df.shape[1] != 5:
#                     #     raise ValueError(f"CSV file {filepath} does not have 5 columns")
#
#                     # Extract label from the folder name
#                     label = os.path.basename(root)
#
#                     # Assign the same index to all rows of this file
#                     index = index_counter
#                     df.index = [index] * len(df)
#
#                     all_dfs.append(df)
#
#                     labels.append([index, label])
#                     index_counter += 1
#
#         all_df = pd.concat(all_dfs, ignore_index=False)
#         labels_df = pd.DataFrame(labels, columns=['index', 'label'])
#
#         # Convert labels to category and get class names
#         labels_df['label'] = labels_df['label'].astype('category')
#         class_names = pd.Index(labels_df['label'].cat.categories)
#
#         # Convert category labels to numerical codes
#         labels_df['label'] = labels_df['label'].cat.codes
#         labels_df.set_index('index', inplace=True)
#
#         all_df.index.name = None
#         labels_df.index.name = None
#
#         return all_df, labels_df, class_names
#
#     def instance_norm(self, case):
#         if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case
#
#     def __getitem__(self, ind):
#         # print('input', torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values).shape,
#         #       torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values).shape)
#         return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
#                torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
#
#     def __len__(self):
#         return len(self.all_IDs)
# old std##############################################################################
# class UEAloader(Dataset):
#     def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
#         self.args = args
#         self.root_path = root_path
#         self.all_df, self.labels_df, self.class_names, self.max_seq_len = self.load_all(root_path, file_list=file_list, flag=flag)
#         self.all_IDs = self.all_df.index.unique()
#
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df
#         print('self.feature_df', len(self.feature_df))
#         # print(self.feature_df)
#         print('self.labels_df', len(self.labels_df))
#         # print(self.labels_df)
#         print(len(self.all_IDs))
#
#     def load_all(self, root_path, file_list=None, flag=None, random_seed=42):
#         all_dfs = []
#         labels = []
#         index_counter = 0
#
#         if flag not in ['TRAIN', 'TEST','TESTALL']:
#             raise ValueError("Flag must be 'TRAIN' or 'TEST'")
#
#         data_path = os.path.join(root_path, flag)
#         # print(f"Data path: {data_path}")  # 添加调试信息
#
#         if not os.path.exists(data_path):
#             raise ValueError(f"Data path {data_path} does not exist")  # 检查路径是否存在
#
#         selected_dirs = ['0', '1','2','3', '4']#['0', '1', '2', '3', '4']
#         max_len = max(
#             get_max_sequence_length(root_path, 'TRAIN', selected_dirs),
#             get_max_sequence_length(root_path, 'TEST', selected_dirs)
#         )
#         # 检查目录内容
#         for root, dirs, files in os.walk(data_path):
#             # print(f"Currently processing directory: {root}")  # 添加调试信息
#             current_dir = os.path.basename(root)
#             if selected_dirs and current_dir not in selected_dirs:
#                 continue
#             for file in files:
#                 # print(f"Found file: {file}")  # 添加调试信息
#                 if file.endswith('.csv'):
#                     filepath = os.path.join(root, file)
#                     df1 = pd.read_csv(filepath, usecols=[0,1,2,3,4], header=0)
#
#                     scaler = MinMaxScaler()
#                     # scaler = StandardScaler()
#                     df = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
#
#                     label = os.path.basename(root)
#
#                     all_dfs.append((df, label, index_counter))
#                     index_counter += 1
#
#         if not all_dfs:
#             raise ValueError("No data found. Please check your data path and selected directories.")
#
#         # 将数据和标签分别存储在列表中
#         data_dfs = []
#         for df, label, index in all_dfs:
#             df.index = [index] * len(df)
#             data_dfs.append(df)
#             labels.append([index, label])
#
#         if not data_dfs:
#             raise ValueError("Dataframes are empty. Check if your data files are correctly read.")
#
#         all_df = pd.concat(data_dfs, ignore_index=False)
#
#         labels_df = pd.DataFrame(labels, columns=['index', 'label'])
#
#         labels_df['label'] = labels_df['label'].astype('category')
#         class_names = pd.Index(labels_df['label'].cat.categories)
#
#         labels_df['label'] = labels_df['label'].cat.codes
#         labels_df.set_index('index', inplace=True)
#
#         all_df.index.name = None
#         labels_df.index.name = None
#
#         return all_df, labels_df, class_names,max_len
#
#     def instance_norm(self, case):
#         if 'EthanolConcentration' in self.root_path:
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case
#
#     def __getitem__(self, ind):
#         return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
#                torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
#
#     def __len__(self):
#          return len(self.all_IDs)
#
# def get_max_sequence_length(root_path, flag,selected_dirs):
#     """
#     获取指定路径中 TRAIN 或 TEST 文件夹内的最长序列长度
#     Args:
#         root_path: 数据集根目录
#         flag: 'TRAIN' 或 'TEST'
#     Returns:
#         max_length: 最长序列的长度
#     """
#     if flag not in ['TRAIN', 'TEST']:
#         raise ValueError("Flag must be 'TRAIN' or 'TEST'")
#
#     data_path = os.path.join(root_path, flag)
#     max_length = 0
#
#     for root, dirs, files in os.walk(data_path):
#         # 0701########################################################3
#         # 提取当前文件夹的名称（最后一级）
#         current_dir = os.path.basename(root)
#         # 如果指定了selected_dirs，并且当前文件夹不在selected_dirs中，则跳过
#         if selected_dirs and current_dir not in selected_dirs:
#             continue
#         ##########################################################################
#         for file in files:
#             if file.endswith('.csv'):
#                 filepath = os.path.join(root, file)
#                 df = pd.read_csv(filepath, header=0)
#                 seq_length = len(df)
#                 if seq_length > max_length:
#                     max_length = seq_length
#
#     return max_length

class UEAloader(Dataset):
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.all_df, self.labels_df, self.class_names, self.max_seq_len,self.filenames_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        # print('self.feature_df', len(self.feature_df))
        # # print(self.feature_df)
        # print('self.labels_df', len(self.labels_df))
        # print(self.max_seq_len)
        # print(len(self.all_IDs))

    def _disentangle(self, df, w, j, save_path=None):
        # 假设输入为 Pandas DataFrame，形状为 [samples, features]

        for col in df.columns:
            # 获取每列数据
            series = df[col].values

            # 小波分解
            coef = pywt.wavedec(series, w, level=j)
            # 分离低频和高频系数
            coefl = [coef[0]]
            for i in range(len(coef) - 1):
                coefl.append(None)

            coefh = [None]
            for i in range(len(coef) - 1):
                coefh.append(coef[i + 1])

            # 小波重构
            xl = pywt.waverec(coefl, w)
            xh = pywt.waverec(coefh, w)
            # 可视化
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(series, label='Original')
            plt.plot(xl, label='Low Frequencies')
            plt.title(f'Original and Low Frequencies for {col}')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(xh, label='High Frequencies', color='r')
            plt.title(f'High Frequencies for {col}')
            plt.legend()
            # 保存图片
            plt.tight_layout()
            plt.savefig(f'{save_path}/{col}_wavelet_decomposition.png')
            plt.close()

    def wavelet_decompose_and_plot_on_single_figure(self,df, w, j, save_path=None):
        """
        对pandas DataFrame中每一列进行小波分解，并将5列数据的分解结果可视化到一张图上后保存为图片。

        参数:
        - data: pd.DataFrame，包含需要分解的数据
        - wavelet: str，小波基函数名称
        - level: int，小波分解层数
        - save_path: str，保存图片的路径

        返回:
        - low_freqs: pd.DataFrame，低频部分的重构结果
        - high_freqs: pd.DataFrame，高频部分的重构结果
        """
        low_freqs = []
        high_freqs = []

        plt.figure(figsize=(14, 10))

        for idx, col in enumerate(df.columns):
            # 获取每列数据
            series = df[col].values

            # 小波分解
            coef = pywt.wavedec(series, w, level=j)

            # 分离低频和高频系数
            coefl = [coef[0]] + [None] * (len(coef) - 1)
            coefh = [None] + coef[1:]

            # 小波重构
            xl = pywt.waverec(coefl, w)
            xh = pywt.waverec(coefh, w)

            # 保存结果
            low_freqs.append(xl)
            high_freqs.append(xh)

            # 绘制原始数据和低频部分
            plt.subplot(5, 3, 3 * idx + 1)
            plt.plot(series, label='Original')
            plt.plot(xl, label='Low Frequencies', color='g')
            plt.title(f'Original and Low Frequencies for {col}')
            plt.legend()
            # 绘制高频部分
            plt.subplot(5, 3, 3 * idx + 2)
            plt.plot(xh, label='High Frequencies', color='r')
            plt.title(f'High Frequencies for {col}')
            plt.legend()
            # 仅绘制低频部分
            plt.subplot(5, 3, 3 * idx + 3)
            plt.plot(xl, label='Low Frequencies', color='g')
            plt.title(f'Low Frequencies Only for {col}')
            plt.legend()

        # 调整布局并保存图片
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def load_all(self, root_path, file_list=None, flag=None, random_seed=42):
        all_dfs = []
        labels = []
        filenames=[]
        index_counter = 0

        if flag not in ['TRAIN', 'TEST', 'TESTALL']:
            raise ValueError("Flag must be 'TRAIN' or 'TEST'")

        data_path = os.path.join(root_path, flag)
        # print(f"Data path: {data_path}")  # 添加调试信息

        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} does not exist")  # 检查路径是否存在

        selected_dirs = ['0', '1','2','3', '4']#['0', '1', '2', '3', '4']
        max_len = max(
            get_max_sequence_length(root_path, 'TRAIN', selected_dirs),
            get_max_sequence_length(root_path, 'TEST', selected_dirs)
        )
        # 检查目录内容
        for root, dirs, files in os.walk(data_path):
            # print(f"Currently processing directory: {root}")  # 添加调试信息
            current_dir = os.path.basename(root)
            if selected_dirs and current_dir not in selected_dirs:
                continue
            for file in files:
                # print(f"Found file: {file}")  # 添加调试信息
                # 如果file是字节流而不是字符串，先解码
                if isinstance(file, bytes):
                    file = file.decode('utf-8')
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    df1 = pd.read_csv(filepath, usecols=self.args.sensornum, header=0)
                    scaler = MinMaxScaler()
                    # scaler = StandardScaler()
                    df = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)

                    # 使用文件名（不含扩展名）作为图片名
                    filename_without_ext = os.path.splitext(file)[0]
                    save_path = f'/data/LJ/dataset/{filename_without_ext}.png'
                    # self._disentangle(df, 'sym2', 2, save_path='/data/LJ/dataset/')
                    # self.wavelet_decompose_and_plot_on_single_figure(df, 'sym2', 2, save_path=save_path)
                    label = os.path.basename(root)

                    all_dfs.append((df, label, index_counter, filename_without_ext))
                    # filenames.append(filename_without_ext)
                    index_counter += 1

        if not all_dfs:
            raise ValueError("No data found. Please check your data path and selected directories.")

        # 将数据和标签分别存储在列表中
        data_dfs = []
        for df, label, index, filename in all_dfs:
            df.index = [index] * len(df)
            data_dfs.append(df)
            labels.append([index, label])
            filenames.append(filename)

        if not data_dfs:
            raise ValueError("Dataframes are empty. Check if your data files are correctly read.")

        all_df = pd.concat(data_dfs, ignore_index=False)

        labels_df = pd.DataFrame(labels, columns=['index', 'label'])
        filenames_df=pd.DataFrame(filenames)
        labels_df['label'] = labels_df['label'].astype('category')
        class_names = pd.Index(labels_df['label'].cat.categories)

        labels_df['label'] = labels_df['label'].cat.codes
        labels_df.set_index('index', inplace=True)

        all_df.index.name = None
        labels_df.index.name = None

        # if flag=='TESTALL':
        #     print(filenames_df)

        return all_df, labels_df, class_names,max_len,filenames_df

    def instance_norm(self, case):
        if 'EthanolConcentration' in self.root_path:
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case
   ############add pathname
    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values),self.filenames_df.loc[self.all_IDs[ind]]

    # def __getitem__(self, ind):
    #     return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
    #            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
         return len(self.all_IDs)

def get_max_sequence_length(root_path, flag,selected_dirs):
    """
    获取指定路径中 TRAIN 或 TEST 文件夹内的最长序列长度
    Args:
        root_path: 数据集根目录
        flag: 'TRAIN' 或 'TEST'
    Returns:
        max_length: 最长序列的长度
    """
    if flag not in ['TRAIN', 'TEST']:
        raise ValueError("Flag must be 'TRAIN' or 'TEST'")

    data_path = os.path.join(root_path, flag)
    max_length = 0

    for root, dirs, files in os.walk(data_path):
        # 0701########################################################3
        # 提取当前文件夹的名称（最后一级）
        current_dir = os.path.basename(root)
        # 如果指定了selected_dirs，并且当前文件夹不在selected_dirs中，则跳过
        if selected_dirs and current_dir not in selected_dirs:
            continue
        ##########################################################################
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath, header=0)
                seq_length = len(df)
                if seq_length > max_length:
                    max_length = seq_length

    return max_length




