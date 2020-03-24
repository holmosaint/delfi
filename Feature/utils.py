import os
import numpy as np
import h5py
import torch
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict


def construct_data_loader(data_path, data_size, batch_size, val_frac=0.2):

    train_size = int(data_size * (1 - val_frac))
    val_size = data_size - train_size

    train_dataset = MyDataset(data_path, 0, train_size)
    if val_size > 0:
        val_dataset = MyDataset(data_path, train_size, val_size)
    else:
        val_dataset = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size)
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size)
    else:
        val_loader = None

    return train_loader, val_loader


class MyDataset(Data.Dataset):

    def __init__(self, data_path, offset, size):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.data = h5py.File(self.data_path, 'r')
        self.offset = offset
        self.size = size

    def __getitem__(self, index):
        Input = self.data.get('trace')[index + self.offset].astype(np.float32)
        try:
            label = self.data.get('label')[index + self.offset]
        except:
            label = None

        return Input, label

    def __len__(self):
        return self.size


def save_checkpoint(store_dir, file_name, epoch, model_state_dict,
                    opt_state_dict, train_loss_list, val_loss_list):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': opt_state_dict,
            'train_loss': train_loss_list,
            'val_loss': val_loss_list
        }, os.path.join(store_dir, file_name))


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)
    """new_checkpoint = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        # print('Name:', k)
        name = k[7:]  # remove 'module.' of dataparallel
        new_checkpoint[name] = v
    checkpoint['model_state_dict'] = new_checkpoint"""
    return checkpoint


def load_data(store_dir, data_file='data.h5', load_dynamic=True, stimuli=False):
    hf = h5py.File(os.path.join(store_dir, data_file), 'r')
    if not stimuli:
        static_arr = hf.get('static_data').value
    else:
        stimuli_arr = hf.get('stimuli_data').value

    if load_dynamic:
        dynamic_arr = hf.get('dynamic_data').value
    else:
        dynamic_arr = None

    target_arr = hf.get('target_data').value

    if not stimuli:
        return static_arr, dynamic_arr, target_arr

    return stimuli_arr, dynamic_arr, target_arr


def init_lstm_hidden(hidden_dim, batch, layer, _cuda):
    document_rnn_init_h = nn.Parameter(nn.init.xavier_uniform(
        torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                       requires_grad=True)
    document_rnn_init_c = nn.Parameter(nn.init.xavier_uniform(
        torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                       requires_grad=True)
    if _cuda:
        document_rnn_init_h = document_rnn_init_h.cuda()
        document_rnn_init_c = document_rnn_init_c.cuda()
    return (document_rnn_init_h, document_rnn_init_c)
