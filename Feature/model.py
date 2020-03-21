import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from resnet import resnet18, resnet34, resnet50, resnet101
from utils import checkpoint, save_checkpoint


class BaseExtractor(ABC):

    def __init__(self):
        pass

    def run_epoch(self, data_loader, opt=None, cuda=True, back=True):
        loss_list = list()
        if back:
            for batch_idx, (Input, label) in enumerate(data_loader):
                if cuda:
                    Input = Input.cuda()
                    label = label.cuda()

                prediction = self.forward(Input)
                loss = self.loss(prediction, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_list.append(loss.item())
        else:
            with torch.no_grad():
                for batch_idx, (Input, label) in enumerate(data_loader):
                    if cuda:
                        Input = Input.cuda()
                        label = label.cuda()

                    prediction = self.forward(Input)
                    loss = self.loss(prediction, label)

                    loss_list.append(loss.item())

        return np.mean(np.array(loss_list))

    def train(self,
              data_loader,
              batch_size,
              epoch,
              val_data_loader=None,
              cuda=True,
              lr=1e-4):
        opt_Adam = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.extractor.net.parameters()),
                                    lr=lr)

        train_loss_list = list()
        val_loss_list = list()

        not_improve = 0
        last_loss = 1e8

        for e in range(epoch):

            train_loss = self.run_epoch(data_loader,
                                        opt_Adam,
                                        cuda=cuda,
                                        back=True)
            train_loss_list.append(train_loss)
            cur_loss = train_loss

            if val_data_loader is not None:
                val_loss = self.run_epoch(val_data_loader,
                                          cuda=cuda,
                                          back=False)
                val_loss_list.append(val_loss)
                cur_loss = val_loss

            print("Epoch {}.\tTrain loss: {}".format(e, train_loss), end='')
            if val_data_loader is not None:
                print("\tVal loss: {}".format(val_loss))
            else:
                print()

            if cur_loss < last_loss:
                not_improve = 0
                last_loss = cur_loss

                save_checkpoint(self.store_path, self.model_name + ".h5", e,
                                self.extractor.net.state_dict(),
                                opt_Adam.state_dict(), train_loss_list,
                                val_loss_list)
            else:
                not_improve += 1

                if not_improve >= 10:
                    print("Early stop at epoch {}".format(e))
                    return

    @abstractmethod
    def loss(self, prediction, label):
        pass

    @abstractmethod
    def get_feature(self, Input):
        pass

    @abstractmethod
    def forward(self, input):
        pass


class PredictiveFeatureExtractor(BaseExtractor):

    def __init__(self, n_segments, res_layers, store_path, input_dim=1):

        self.model_name = 'predictive'
        self.store_path = store_path

        self.n_segments = n_segments
        self.res_layers = res_layers
        self.input_dim = input_dim

        self.net = PredctiveNeuralNetowork(n_segments, res_layers, input_dim)

        self.loss_func = nn.NLLLoss()

    def loss(self, prediction, label):
        return self.loss_func(prediction, label)

    def forward(self, Input):
        return self.net(Input)

    def get_feature(self, input):
        raise NotImplementedError


class PredctiveNeuralNetowork(nn.Module):

    def __init__(self, n_segments, res_layers, input_dim):
        self.n_segments = n_segments
        self.res_layers = res_layers
        self.input_dim = input_dim

        if self.res_layers == 18:
            self.extractor = resnet18(input_dim)
        elif self.res_layers == 34:
            self.extractor = resnet34(input_dim)
        elif self.res_layers == 50:
            self.extractor = resnet50(input_dim)
        elif self.res_layers == 101:
            self.extractor = resnet101(input_dim)

        self.fc = nn.Sequential(*[
            nn.Linear(512, self.n_segments),
            nn.LogSoftmax(dim=-1),
        ])

    def forward(self, Input):
        x = self.extractor(Input)
        x = self.fc(x)
        return x
