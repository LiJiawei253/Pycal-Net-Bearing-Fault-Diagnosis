import os

import numpy as np
import sklearn.model_selection
import torch
from acoustics import generator as ag
from scipy.io import loadmat
from sklearn import preprocessing
from tqdm import tqdm

from utils.noise_observe import add_audio_noise

HBdata = ['K001', "K002", 'K003', 'K004', 'K005', 'K006']
RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']


def Paderborn_Processing(
    file_path,
    load,
    length=2048,
    use_sliding_window=True,
    step_size=2048,
    sample_number=20,
    normal=True,
    noise='Gaussian',
    snr=-6,
):
    train_path = os.path.join(file_path, load)   # 训练集样本路径

    # 获得训练集文件夹下所有.mat文件名
    train_filenames = os.listdir(train_path)
    train_filenames = [i for i in train_filenames if i.endswith('.mat')]

    def wgn(x, snr_value):
        snr_lin = 10 ** (snr_value / 10.0)
        xpower = np.sum(np.absolute(x) ** 2, axis=1) / x.shape[0]
        npower = xpower / snr_lin
        npower = np.repeat(npower.reshape(-1, 1), x.shape[1], axis=1)
        return np.random.standard_normal(x.shape) * np.sqrt(npower)

    def add_noise(data, snr_num):
        rand_data = wgn(data, snr_num)
        n_data = data + rand_data
        return n_data, rand_data

    def laplace_noise(x, snr_value):
        p_signal = np.sum(np.abs(x) ** 2, axis=1) / len(x)
        p_noise = p_signal / (10 ** (snr_value / 10))
        p_noise = np.repeat(p_noise.reshape(-1, 1), x.shape[1], axis=1)
        white_noise = np.random.laplace(size=x.shape) * np.sqrt(p_noise)
        return x + white_noise, x

    def pink_noise(x, snr_value):
        p_signal = np.sum(np.abs(x) ** 2, axis=1) / len(x)
        p_noise = p_signal / (10 ** (snr_value / 10))
        p_noise = np.repeat(p_noise.reshape(-1, 1), x.shape[1], axis=1)
        n = ag.noise(x.shape[0] * x.shape[1], color='pink').reshape(x.shape[0], x.shape[1]) * np.sqrt(p_noise)
        return x + n, x

    def capture(path, filenames):
        """
        读取 paderborn 数据集中的振动数据(vibration_1)
        :return: 每个文件的振动数据(dict)
        """
        data = {}
        for fname in filenames:
            file_path_ = os.path.join(path, fname)
            file = loadmat(file_path_)
            file_keys = fname.strip('.mat')
            for j in file[file_keys][0][0]:
                if 'Name' in str(j.dtype):
                    if 'vibration_1' in j[0]['Name']:
                        index = np.argwhere(j[0]['Name'] == 'vibration_1')
                        data[file_keys] = j[0]['Data'][index][0][0][0]
        return data

    def slice_samples(data, samp_num):
        data_keys = data.keys()
        Data_Samples, Labels = [], []
        Test_Samples, Test_Labels = [], []

        for key in tqdm(list(data_keys)):
            slice_data = data[key]
            start = 0
            end_index = len(slice_data)

            if 'K0' in key:
                class_num = 0
            elif 'KA04' in key:
                class_num = 1
            elif 'KA15' in key:
                class_num = 2
            elif 'KA16' in key:
                class_num = 3
            elif 'KA22' in key:
                class_num = 4
            elif 'KA30' in key:
                class_num = 5
            elif 'KB23' in key:
                class_num = 6
            elif 'KB24' in key:
                class_num = 7
            elif 'KB27' in key:
                class_num = 8
            elif 'KI04' in key or 'KI14' in key:
                class_num = 9
            elif 'KI16' in key:
                class_num = 10
            elif 'KI17' in key:
                class_num = 11
            elif 'KI18' in key:
                class_num = 12
            elif 'KI21' in key:
                class_num = 13
            else:
                continue

            samp_num1 = len(slice_data) // step_size if key[-2:] == "20" else samp_num
            for _ in range(samp_num1):
                if use_sliding_window:
                    sample = slice_data[start:start + length]
                    start = start + step_size
                else:
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]

                if key[-2:] == "20":
                    Test_Samples.append(sample)
                    Test_Labels.append(class_num)
                else:
                    Data_Samples.append(sample)
                    Labels.append(class_num)

        return (
            np.array(Data_Samples),
            np.array(Labels),
            np.array(Test_Samples),
            np.array(Test_Labels),
        )

    def scalar_stand(Train_X, Val_X, Test_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        return scalar.transform(Train_X), scalar.transform(Val_X), scalar.transform(Test_X)

    train_data = capture(train_path, train_filenames)
    Train_X, Train_Y, Test_X, Test_Y = slice_samples(train_data, sample_number)

    Train_X, Val_X, Train_Y, Val_Y = sklearn.model_selection.train_test_split(
        Train_X, Train_Y, train_size=0.8, test_size=0.2
    )

    if noise == 'Gaussian':
        Train_X, _ = add_noise(Train_X, snr)
        Val_X, _ = add_noise(Val_X, snr)
        Test_X, _ = add_noise(Test_X, snr)
    elif noise == 'pink':
        Train_X, _ = pink_noise(Train_X, snr)
        Val_X, _ = pink_noise(Val_X, snr)
        Test_X, _ = pink_noise(Test_X, snr)
    elif noise == 'Laplace':
        Train_X, _ = laplace_noise(Train_X, snr)
        Val_X, _ = laplace_noise(Val_X, snr)
        Test_X, _ = laplace_noise(Test_X, snr)
    elif noise == 'airplane':
        n = np.load('data/audio/airplanenoise1e-10.npy')
        Train_X = add_audio_noise(n, Train_X)
        Val_X = add_audio_noise(n, Val_X)
        Test_X = add_audio_noise(n, Test_X)
    elif noise == 'truck':
        n = np.load('data/audio/trucknoise1e-10.npy')
        Train_X = add_audio_noise(n, Train_X)
        Val_X = add_audio_noise(n, Val_X)
        Test_X = add_audio_noise(n, Test_X)

    if normal:
        Train_X, Val_X, Test_X = scalar_stand(Train_X, Val_X, Test_X)

    Train_X, Val_X, Test_X = Train_X[:, np.newaxis, :], Val_X[:, np.newaxis, :], Test_X[:, np.newaxis, :]

    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Val_X = torch.tensor(Val_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Val_Y = torch.tensor(Val_Y, dtype=torch.float)
    Test_Y = torch.tensor(Test_Y, dtype=torch.long)

    return Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y


if __name__ == '__main__':
    file_path = '../data/Paderborn'
    load = 'N15_M07_F10'
    Paderborn_Processing(file_path=file_path, load=load, noise='airplane')

