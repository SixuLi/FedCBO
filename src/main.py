import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import logging

import torch.cuda
from sklearn.cluster import KMeans


import src.init as init
from src.FedCBO import FedCBO_NN




parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='test')
parser.add_argument('--model_name', type=str, default='FC')
parser.add_argument('--result_path', type=str, default='results')
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--data_name', type=str)
parser.add_argument('--is_data', default=False, action='store_true')
parser.add_argument('--is_subset', default=False, action='store_true')

parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--p', type=int, default=2, help='Number of clusters.')
parser.add_argument('--N', type=int, default=100, help='Total number of agents.')
parser.add_argument('--M', type=int, default=20, help='Number of agents involved in one round federated updating.')
parser.add_argument('--n', type=int, default=10000, help='Total number of data points.')
parser.add_argument('--m', type=int, default=50, help='Batch size of data points used for training.')
parser.add_argument('--T', type=int, default=100, help='Total time step.')
parser.add_argument('--Lambda', type=int, default=1)
parser.add_argument('--Sigma', type=float, default=5)
parser.add_argument('--Alpha', type=float, default=30)
parser.add_argument('--Gamma', type=float, default=0.01)
parser.add_argument('--Epsilon', type=float, default=1e-3, help='Threshold of stopping criterion.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Probability of random selection.')
parser.add_argument('--epsilon_decay', type=float, default=0.01)
parser.add_argument('--epsilon_threshold', type=float, default=0.1)
parser.add_argument('--annealing', default=False, action='store_true')
parser.add_argument('--d', type=int, default=1, help='Dimension of model parameters.')

parser.add_argument('--input_dim', type=int, default=784)
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--bias', default=False, action='store_true')

parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--local_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'MultiStepLR'])
parser.add_argument('--lr_step_size', type=int, default=10000)
parser.add_argument('--lr_gamma', type=float, default=1.0)
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--is_communication', default=False, action='store_true')

parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--load_checkpoint', default=False, action='store_true')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info("Running training for {}".format(args.experiment_name))
logging.info("Seed {}".format(args.seed))

train_init = init.Init(args=args)

os.environ['CUDA_VISBLE_DEVICES'] = args.gpu_ids
args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
args.cuda = not args.no_cuda and torch.cuda.is_available()
FedCBO = FedCBO_NN(train_init=train_init, args=args)
if args.is_communication:
    FedCBO.train_with_comm()
else:
    FedCBO.train_without_comm()
logging.info('Average acc of local agents: {}'.format(torch.sum(FedCBO.store_test_acc) / FedCBO.store_test_acc.size(0)))


