import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import XRFDataset, iDataset


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)                   # 设置数据集
        assert init_cls <= self.total_class_num, "No enough classes."
        self._increments = [init_cls]                                   # self._increments = [15, 10, 10, 10, 10]
        while sum(self._increments) + increment < self.total_class_num:
            self._increments.append(increment)
        offset = self.total_class_num - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return self.total_class_num

    def get_dataset(
            self, indices, source, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
            is_train = True
        elif source == "test":
            x, y = self._test_data, self._test_targets
            is_train = False
        else:
            raise ValueError("Unknown data source {}.".format(source))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
        # print("Data:",data) # data格式：【【类别一的数据】，【类别二的数据】，...】
        dataset = iDataset(data, is_train=is_train)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            dataset._construct_exemplar(appendent_data, appendent_targets)

        if ret_data:
            return dataset.data, dataset.label, dataset
        else:
            return dataset

    def get_dataset_with_split(
            self, indices, source, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
            is_train = True
        elif source == "test":
            x, y = self._test_data, self._test_targets
            is_train = False
        else:
            raise ValueError("Unknown data source {}.".format(source))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(list(np.array(class_data)[val_indx]))
            val_targets.append(list(np.array(class_targets)[val_indx]))
            train_data.append(list(np.array(class_data)[train_indx]))
            train_targets.append(list(np.array(class_targets)[train_indx]))

        train_dataset = iDataset(train_data, is_train=is_train)
        val_dataset = iDataset(val_data, is_train=is_train)
        train_edata, train_etargets = [], []
        val_edata, val_etargets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_edata.extend(list(np.array(append_data)[val_indx]))
                val_etargets.extend(list(np.array(append_targets)[val_indx]))
                train_edata.extend(list(np.array(append_data)[train_indx]))
                train_etargets.extend(list(np.array(append_targets)[train_indx]))
            train_dataset._construct_exemplar(train_edata, train_etargets)
            val_dataset._construct_exemplar(val_edata, val_etargets)

        return train_dataset, val_dataset


    def _setup_data(self, dataset_name, shuffle, seed):
        train_dataset, test_dataset, total_class_num = _get_idata(dataset_name)

        # Data
        self._train_data, self._train_targets = train_dataset.data['file_name'], train_dataset.data['label']
        self._test_data, self._test_targets = test_dataset.data['file_name'], test_dataset.data['label']
        self.total_class_num = total_class_num

    def _select(self, x, y, low_range, high_range):
        # 按类别返回列表
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        new_x = []
        new_y = []
        for idx in idxes:
            new_x.append(x[idx])
            new_y.append(y[idx])
        return new_x, new_y

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


def _get_idata(dataset_name):
    # 获取数据集
    name = dataset_name.lower()
    if name == "xrfdataset":
        return XRFDataset(is_train=True), XRFDataset(is_train=False), 55
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
