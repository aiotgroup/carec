import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from sklearn.decomposition import PCA
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

EPSILON = 1e-8


init_epoch = 1
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 1
lrate = 0.1
milestones = [60, 100, 140]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class WA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.args = args

    def after_task(self):
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes - self._known_classes)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        #self.save_checkpoint("{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        
        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            appendent=self._get_memory(),
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        #self.plot_tsne(self.args,data_manager)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            #if len(self._multiple_gpus) > 1:
            #    self._network.module.weight_align(
            #        self._total_classes - self._known_classes
            #    )
            #else:
            #    self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        kd_lambda = self._known_classes / self._total_classes
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_clf, losses_kd = 0., 0.
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                #loss_kd = 0.01*nn.MSELoss()(logits[:,:self._known_classes], self._old_network(inputs)["logits"])
                loss_kd = _KD_loss(
                   logits[:, : self._known_classes],
                   self._old_network(inputs)["logits"],
                   T,
                )

                loss = (1-kd_lambda) * loss_clf + kd_lambda * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf/len(train_loader), 
                    losses_kd/len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf/len(train_loader), 
                    losses_kd/len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def to_categorical(self,y, num_classes=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def plot_tsne(self,args,data_manager):
        # test_loader = self.test_loader
        model = IncrementalNet(args, False)
        model.update_fc(self._total_classes)
        model.load_state_dict(torch.load("{}_{}_{}_{}.pkl".format(self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test",
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=True, num_workers=0
        )

        features = []
        targets = []
        for (inpu, target) in test_loader:
            temp = model(inpu)['logits']
            # print(temp.detach().numpy().shape)
            if features == []:
                features = temp.detach().numpy()
                targets = target.detach().numpy()
                continue
            features = np.concatenate((features,temp.detach().numpy()),axis=0)
            targets = np.concatenate((targets,target.detach().numpy()),axis=0)
        print(features.shape)
        #-------------------------------PCA,tSNE降维分析--------------------------------
        pca = PCA(n_components=15+(self._cur_task*10))# 总的类别
        pca_result = pca.fit_transform(features)
        print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

        #Run T-SNE on the PCA features.
        tsne = TSNE(n_components=2, verbose = 1 )#,perplexity=50, learning_rate=200, n_iter=1000)
        tsne_results = tsne.fit_transform(pca_result[:2000])
        #------------------------------可视化--------------------------------
        y_test_cat = self.to_categorical(targets[:2000], num_classes = 15+self._cur_task*10)# 总的类别
        color_map = np.argmax(y_test_cat, axis=1)
        plt.figure(figsize=(10,10))
        color_list = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE',
              '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3', '#808000', '#FFD8B1', '#000080', '#808080',
              '#FFFFFF', '#000000', '#FF00FF', '#FF7F00', '#FFD700', '#00FF00', '#00FFFF', '#FF0000', '#8B4513', '#00CED1',
              '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#228B22', '#FFFAF0', '#DCDCDC', '#F8F8FF',
              '#FF4500', '#DA70D6', '#DAA520', '#FF8C00', '#FA8072', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
              '#D2691E', '#FF69B4', '#8B008B', '#ADFF2F', '#F0E68C', '#CD5C5C', '#4B0082']

        for cl in range(15+self._cur_task*10):# 总的类别
            indices = np.where(color_map==cl)
            indices = indices[0]
            #plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], c=targets[:600], label=cl,cmap = ListedColormap(colors))
            plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl,c=color_list[cl % len(color_list)])
        
        plt.legend()
        plt.savefig("checkpoint/tsne_wa_"+str(self._cur_task)+".jpg")

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
