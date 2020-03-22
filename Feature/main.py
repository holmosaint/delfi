import numpy as np
import torch
import argparse

from model import TimeContrastiveFeatureExtractor
from utils import construct_data_loader


def train_time_contrastive(args):
    n_segments = args.n_segments
    store_path = args.store_path
    res_layers = args.res_layers
    input_dim = args.input_dim
    data_file = args.data
    batch_size = args.batch
    data_size = args.size
    epoch = args.epoch
    cuda = args.cuda
    lr = args.lr

    time_contrastive_feature_extractor = TimeContrastiveFeatureExtractor(
        n_segments, res_layers, store_path, input_dim)

    train_loader, val_loader = construct_data_loader(data_file, data_size,
                                                     batch_size)

    time_contrastive_feature_extractor.train(train_loader,
                                             batch_size,
                                             epoch,
                                             val_data_loader=val_loader,
                                             cuda=cuda,
                                             lr=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data',
                        help='Path to the data h5 file',
                        type=str,
                        required=True)
    parser.add_argument('-size', help='Number of data', type=int, required=True)
    parser.add_argument('-batch', help='Batch size', type=int, default=32)
    parser.add_argument('-epoch', help='Epoch', type=int, default=100)
    parser.add_argument('-cuda', help='Cuda', type=bool, default=True)
    parser.add_argument('-lr', help='Learning rate', type=float, default=1e-4)

    subparser = parser.add_subparsers(description="[time_contrastive]")

    time_contrastive_parser = subparser.add_parser('time_contrastive')
    time_contrastive_parser.add_argument('-n_segments',
                                         help='Number of segments to classify',
                                         type=int,
                                         required=True)
    time_contrastive_parser.add_argument(
        '-store_path',
        help='Path to store the neural network model',
        type=str,
        required=True)
    time_contrastive_parser.add_argument('-res_layers',
                                         help='ResNet layers [18, 34, 50, 101]',
                                         type=int,
                                         default=18)
    time_contrastive_parser.add_argument('-input_dim',
                                         help='Input dimension',
                                         type=int,
                                         default=1)
    time_contrastive_parser.set_defaults(func=train_time_contrastive)

    args = parser.parse_args()
    args.func(args)
