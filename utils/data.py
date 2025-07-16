import logging
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

log = logging.getLogger(__name__)


class iDataset(Dataset):
    '''最终用于训练的Dataset'''
    def __init__(self, data_list=[], file_path='/home/dataset/XRFDataset/', is_train=True):
        super(iDataset, self).__init__()
        self.data = []
        self.label = []
        self.is_train = is_train
        
        for cname in data_list:
            for string in cname:
                tmpt = string
                tmpl = int(string.split('_')[1]) - 1
                self.data.append(tmpt)
                self.label.append(tmpl)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        wifi_data = load_wifi(data, self.is_train)
        label = self.label[idx]
        return wifi_data, label
    
    def _construct_exemplar(self, edata, elabel):
        self.data.extend(edata)     
        self.label.extend(elabel)

class XRFDataset(Dataset):
    '''按照文件名实时加载npy'''
    def __init__(self, file_path='/home/dataset/XRFDataset/', is_train=True):
        super(XRFDataset, self).__init__()
        self.file_path = file_path
        self.is_train = is_train
        if self.is_train:
            self.file = '/home/dataset/XRFDataset/train_0.7.txt'   #'/home/fuqunhang/CILProject/XRFDATA/ra_train_0.7.txt'
        else:
            self.file = '/home/dataset/XRFDataset/val_0.7.txt'   #'/home/fuqunhang/CILProject/XRFDATA/ra_val_0.7.txt'
        file = open(self.file)
        val_list = file.readlines()
        self.data = {
            'file_name': list(),
            'label': list(),
        }
        for string in val_list:
            tmpt = string.split(',')[0]
            self.data['file_name'].append(tmpt)
            tmpl = int(string.split(',')[1]) - 1
            self.data['label'].append(tmpl)
        log.info("加载完毕XRF原始数据集")

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        wifi_data = load_wifi(file_name, self.is_train, path='/home/dataset/XRFDataset/new_data/')
        return wifi_data, label


def load_wifi(filename, is_train, path='/home/dataset/XRFDataset/new_data/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_mmwave(filename, is_train, path='/home/dataset/XRFDataset/new_data/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    # 读取mat文件
    mmWave_data = np.load(path + 'mmWave/' + filename + ".npy")
    mmWave_data = mmWave_data.squeeze()
    return torch.from_numpy(mmWave_data).float()


def load_rfid(filename, is_train, path='/home/dataset/XRFDataset/new_data/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'RFID/' + filename + ".npy")
    return torch.from_numpy(record).float()