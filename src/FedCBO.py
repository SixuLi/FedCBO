import numpy as np
import math
import logging
import tqdm
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os

import torch.utils.data
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple

from src.model import FCModel
from src.dataset import rotatedMNIST
from src.utils import average_meter
from src.utils.util import chunkify


from scipy.linalg import block_diag
import pdb


def get_optimizer(args, model):
    logging.info('Optimizer is {}'.format(args.optimizer))
    if args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               momentum=args.momentum)
    else:
        raise NotImplementedError

def get_lr_scheduler(args, optimizer):
    logging.info('LR Scheduler is {}'.format(args.lr_scheduler))
    if args.lr_scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                               gamma=args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones,
                                                    gamma=args.lr_gamma)
    else:
        raise NotImplementedError

class FedCBO_NN:
    """
    Hyperparameters:
    N: Number of agents
    M: Number of agents being sampled to involve in federated training process
    T: Total time steps
    input_dim: Input dimension of NN
    hidden_dims: Hidden layers' dimensions of NN
    output_dim: Output dimension of NN
    Lambda, Sigma, Alpha: Hyparameters control SDEs
    Gamma: Local aggregation step size
    """
    def __init__(self, train_init, args):
        self.train_init = train_init
        self.args = args
        self.N = args.N
        self.M = args.M
        self.T = args.T
        self.Lambda = args.Lambda
        self.Sigma = args.Sigma
        self.Alpha = args.Alpha
        self.Gamma = args.Gamma
        self.seed = args.seed
        self.data_name = args.data_name
        self.epsilon = args.epsilon

        self.input_dim = args.input_dim
        self.hidden_dims = args.hidden_dims
        self.outdim = args.output_dim
        self.bias = args.bias

        self.setup_datasets()

        self.initialization()
        self.store_test_acc = torch.zeros(len(self.agents_idx))
        self.test_acc_after_local_sgd = torch.zeros(len(self.agents_idx))

    def initialization(self):
        """
        Initialize the local nns.
        """

        # Initialize the models and sampling likelihood matrix
        if self.args.load_checkpoint:
            checkpoint_path = os.path.join(self.train_init.output_path, 'models.pt')
            checkpoint = torch.load(checkpoint_path)
            self.agents = checkpoint['models']
            self.sampling_likelihood = checkpoint['sampling_likelihood'].numpy()
            self.epsilon = checkpoint['epsilon']
            self.starting_epoch = checkpoint['epoch']
            logging.info('starting_epsilon: {}'.format(self.epsilon))
            logging.info('starting_epoch: {}'.format(self.starting_epoch))

            if torch.cuda.is_available():
                for model in self.agents:
                    model.cuda()

        else:
            self.agents = []
            for _ in range(self.args.N):
                model = FCModel(input_dim=self.input_dim, hidden_dims=self.hidden_dims,
                                output_dim=self.outdim, bias=self.bias)
                if torch.cuda.is_available():
                    model.cuda()
                self.agents.append(model)

            self.sampling_likelihood = np.ones((self.N, self.N))
            np.fill_diagonal(self.sampling_likelihood, -np.inf)

        self.agents_idx = np.arange(0, self.N, 1)

    def setup_datasets(self):
        np.random.seed(self.args.seed)

        # Generate indices for each dataset, also write cluster info

        self.dataset = {}

        train_data = []
        test_data = []
        for cluster_idx in range(self.args.p):
            if self.args.data_name == 'rotated_mnist':
                transform = transforms.ToTensor()
                trainset = rotatedMNIST(self.args, train=True, download=True,
                                          transform=transform, cluster_idx=cluster_idx, is_subset=self.args.is_subset)
                testset = rotatedMNIST(self.args, train=False, download=True,
                                          transform=transform, cluster_idx=cluster_idx)
            train_data.append(trainset)
            test_data.append(testset)

        self.num_local_data = len(trainset) * self.args.p // self.args.N

        train_dataset = {}
        train_dataset['data_indices'], train_dataset['cluster_assign'] = \
            self._setup_dataset(len(trainset), self.args.p, self.args.N, self.num_local_data)
        train_dataset['data'] = train_data
        self.dataset['train'] = train_dataset


        test_dataset = {}
        test_dataset['data'] = test_data
        self.dataset['test'] = test_dataset

    def _setup_dataset(self, num_data, p, N, num_local_data, random=True):
        # assert (N // p) * num_local_data == num_data

        data_indices = []
        cluster_assign = []
        agents_per_cluster = N // p

        for p_i in range(p):
            if random:
                ll = list(np.random.permutation(num_data))
            else:
                ll = list(range(num_data))

            ll2 = chunkify(ll, agents_per_cluster)  # splits ll into N lists with size num_local_data
            data_indices += ll2

            cluster_assign += [p_i for _ in range(agents_per_cluster)]

        data_indices = np.array(data_indices)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == N

        self.cluster_assign = cluster_assign

        return data_indices, cluster_assign

    def get_agent_dataloader(self, agent_idx, train=True):
        cluster_idx = self.dataset['train']['cluster_assign'][agent_idx]
        if train:
            dataset = self.dataset['train']
            data_indices = dataset['data_indices'][agent_idx]

            data = dataset['data'][cluster_idx]
            images, targets = data.data[data_indices].float(), data.targets[data_indices]
        else:
            dataset = self.dataset['test']
            data = dataset['data'][cluster_idx]
            images, targets = data.data.float(), data.targets

        local_data = torch.utils.data.TensorDataset(images, targets)
        dataloader = torch.utils.data.DataLoader(local_data, batch_size=self.args.batch_size, shuffle=True)

        return dataloader



    def weighted_avg_single_layer(self, thetas, mu, layer):
        # weighted average the parameters for given layer
        avg_weight = None
        avg_bias = None
        with torch.no_grad():
            for j, nn in enumerate(thetas):
                 if avg_weight is None:
                    avg_weight = nn.get_layer_weights(layer_num=layer) * mu[j]
                 else:
                    avg_weight += nn.get_layer_weights(layer_num=layer) * mu[j]
                 if avg_bias is None:
                    avg_bias = nn.get_layer_bias(layer_num=layer) * mu[j]
                 else:
                    avg_bias += nn.get_layer_bias(layer_num=layer) * mu[j]
            avg_weight /= torch.sum(mu)
            avg_bias /= torch.sum(mu)
        return (avg_weight, avg_bias)

    def local_aggregation(self, is_first_round=False):
        cur_agents = []
        for agent_idx in self.agents_idx:
            logging.info('Start local aggregation for agent {}'.format(agent_idx + 1))

            # Select other agents
            A_t = self.agents_selection(agent_idx=agent_idx, is_first_round=is_first_round)
            thetas = np.take(self.agents, A_t, axis=0)

            # Check the correctness of selecting process
            same_cluster_agents = np.where(self.dataset['train']['cluster_assign'][A_t] ==
                                           self.dataset['train']['cluster_assign'][agent_idx])[0]
            # logging.info(
            #     'Num of agents in same cluster / Num of selected agents: {}/{}'.format(same_cluster_agents.size,
            #                                                                            A_t.size))
            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write(
                    'Num of agents in same cluster / Num of selected agents: {}/{}\n'.format(same_cluster_agents.size,
                                                                                             A_t.size))

            theta_p = copy.deepcopy(self.agents[agent_idx])
            train_acc_p, train_loss_p = self.evaluate(model_idx=agent_idx, cur_agent_idx=agent_idx, tag='train')

            mu = torch.zeros(A_t.size)
            for i, model in enumerate(thetas):
                train_acc, train_loss = self.evaluate(model_idx=A_t[i], cur_agent_idx=agent_idx, tag='train')

                # Update sampling likelihood
                self.sampling_likelihood[agent_idx][A_t[i]] += train_loss_p.sum - train_loss.sum

                # Calculate weight
                mu_i = torch.exp(torch.tensor([-self.Alpha]) * train_loss.sum)
                mu[i] = mu_i
            if torch.cuda.is_available():
                mu.cuda()
            # print(mu)

            # Local aggregation layer by layer
            for i in range(1, theta_p.num_layers + 1):
                # calculate the consensus point for layer i
                mu_p_weight, mu_p_bias = self.weighted_avg_single_layer(thetas=thetas, mu=mu, layer=i)

                # Update theta_p for layer i
                target_weight = theta_p.get_layer_weights(layer_num=i)
                target_bias = theta_p.get_layer_bias(layer_num=i)

                target_weight.data = target_weight.data - self.Lambda * self.Gamma * (target_weight.data - mu_p_weight)
                target_bias.data = target_bias.data - self.Lambda * self.Gamma * (target_bias.data - mu_p_bias)

            cur_agents.append(theta_p)

        for agent_idx in self.agents_idx:
            self.agents[agent_idx] = copy.deepcopy(cur_agents[agent_idx])
            test_acc, test_loss = self.evaluate(model_idx=agent_idx, cur_agent_idx=agent_idx, tag='test')
            self.store_test_acc[agent_idx] = test_acc
            logging.info('Agent {} acc test: {}'.format(agent_idx + 1, test_acc))

    def agents_selection(self, agent_idx, is_first_round=False):
        logging.info('Start agents selections for local agent {}'.format(agent_idx+1))
        sampling_likelihood = copy.deepcopy(self.sampling_likelihood[agent_idx])

        if is_first_round:
            A_t = np.random.choice(np.where(self.agents_idx != agent_idx)[0], self.M, replace=False)
        else:
            num_rand_sample = int(self.epsilon * self.M)

            # Randomly pick num_rand_sample number of agents
            rand_A_t = np.random.choice(np.where(self.agents_idx != agent_idx)[0] , num_rand_sample, replace=False)
            # print(rand_A_t)

            # Sample the remaining agents through the order of sampling likelihood
            sampling_likelihood[rand_A_t] = -np.inf
            num_remaining_sample = self.M - num_rand_sample
            remaining_A_t = sampling_likelihood.argsort()[-num_remaining_sample:]
            A_t = np.concatenate((rand_A_t, remaining_A_t))

        return A_t



    def evaluate(self, model_idx, cur_agent_idx, tag='train'):
        # Using model_idx to load the nn we are going to evaluate.
        # Using cur_agent_idx to load the dataset we are going to use.

        model = self.agents[model_idx]
        if tag == 'train':
            dataloader = self.get_agent_dataloader(agent_idx=cur_agent_idx, train=True)
        elif tag == 'test':
            dataloader =self.get_agent_dataloader(agent_idx=cur_agent_idx, train=False)
        else:
            raise NotImplementedError
        model.eval()

        total = 0
        correct = 0
        loss_logger = average_meter.AverageMeter()

        for batch_idx, (images, labels) in enumerate(dataloader):
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.model_name == 'FC':
                logits = model(images.view(images.size(0), -1))
            else:
                logits = model(images)
            loss = F.cross_entropy(logits, labels)
            prediction = torch.argmax(logits, dim=1)
            total += images.size(0)
            correct += torch.sum(labels == prediction)
            loss_logger.update(loss.item())

        accuracy = 100.0 * correct / total
        return accuracy, loss_logger


    def local_sgd(self, agent_idx, epochs):
        model = self.agents[agent_idx]
        dataloader = self.get_agent_dataloader(agent_idx=agent_idx, train=True)

        logging.info('Start local SGD training.')
        optimizer = get_optimizer(self.args, model)
        lr_scheduler = get_lr_scheduler(self.args, optimizer)

        for epoch in range(1, epochs + 1):
            logging.info('Epoch {}/{}'.format(epoch, epochs))

            model.train()
            tbar = tqdm.tqdm(dataloader)
            loss_logger = average_meter.AverageMeter()

            # Mini-batch sgd
            for batch_idx, (images, labels) in enumerate(tbar):
                # images = images.float()
                if self.args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                if self.args.model_name == 'FC':
                    logits = model(images.view(images.size(0), -1))
                else:
                    logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss_logger.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

        test_acc, test_loss = self.evaluate(model_idx=agent_idx, cur_agent_idx=agent_idx, tag='test')
        self.test_acc_after_local_sgd[agent_idx] = test_acc

        logging.info('Local SGD training finished.')
        logging.info('Acc test:{}'.format(test_acc))

    def train_with_comm(self):
        if self.args.load_checkpoint:
            t = self.starting_epoch
            is_first_round = False
        else:
            t = 0
            is_first_round = True
        while t < self.T:
            if t % 10 == 0:
                logging.info('Training epoch {}'.format(t))

            # Local update for each agent
            for agent_idx in self.agents_idx:
                logging.info('Training model for agent {} with local sgd'.format(agent_idx + 1))
                self.local_sgd(agent_idx=agent_idx, epochs=self.args.local_epochs)

            # Local aggregation at time step t
            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('Communication round: {} \n'.format(t))
            self.local_aggregation(is_first_round=is_first_round)
            is_first_round = False

            # epsilon decay
            if self.epsilon > self.args.epsilon_threshold:
                self.epsilon -= self.args.epsilon_decay

            t += 1
            self.save_checkpoint(epoch=t)
            with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
                f.write('Communication round: {} \n'.format(t))
                f.write('Average acc of local agents: {} \n'.format(
                    torch.sum(self.store_test_acc) / self.store_test_acc.size(0)))

        logging.info("Training finished at epoch {}".format(t))
        for agent_idx in self.agents_idx:
            test_acc, test_loss = self.evaluate(model_idx=agent_idx, cur_agent_idx=agent_idx, tag='test')
            self.store_test_acc[agent_idx] = test_acc

            logging.info('Agent {} acc test:{}'.format(agent_idx + 1, test_acc))

        f.write('Final result: \n')
        f.write('Average acc of local agents: {} \n'.format(torch.sum(self.store_test_acc) / self.store_test_acc.size(0)))

    def train_without_comm(self):
        for i, agent_idx in enumerate(self.agents_idx):
            logging.info('Training model for agent {} with local SGD'.format(i + 1))
            self.local_sgd(agent_idx=agent_idx, epochs=int(self.args.local_epochs * self.T))
            test_acc, test_loss = self.evaluate(model_idx=agent_idx, cur_agent_idx=agent_idx, tag='test')
            self.store_test_acc[agent_idx] = test_acc
            logging.info('Agent {} acc test:{}'.format(agent_idx+1, test_acc))

        with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
            f.write('Final result: \n')
            f.write('Average acc of local agents: {} \n'.format(torch.sum(self.store_test_acc) / self.store_test_acc.size(0)))

    def save_checkpoint(self, epoch):
        sampling_lilelihood = torch.from_numpy(self.sampling_likelihood)
        torch.save({'models': self.agents,
                    'sampling_likelihood': sampling_lilelihood,
                    'epoch': epoch,
                    'epsilon': self.epsilon},
                   os.path.join(self.train_init.output_path, 'models.pt'))






            
        

if __name__ == '__main__':
    # Test FedCBO
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='FC')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--is_data', default=False, action='store_true')

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--p', type=int, default=2, help='Number of clusters.')
    parser.add_argument('--N', type=int, default=10, help='Total number of agents.')
    parser.add_argument('--M', type=int, default=5, help='Number of agents involved in one round federated updating.')
    parser.add_argument('--n', type=int, default=10000, help='Total number of data points.')
    parser.add_argument('--m', type=int, default=50, help='Batch size of data points used for training.')
    parser.add_argument('--T', type=int, default=100, help='Total time step.')
    parser.add_argument('--Lambda', type=int, default=1)
    parser.add_argument('--Sigma', type=float, default=5)
    parser.add_argument('--Alpha', type=float, default=30)
    parser.add_argument('--Gamma', type=float, default=0.01)
    parser.add_argument('--Epsilon', type=float, default=1e-3, help='Threshold of stopping criterion.')
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

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')

    args = parser.parse_args()

    FedCBO = FedCBO_NN(train_init=None, args=args)

    dataloader = FedCBO.get_agent_dataloader(agent_idx=2, train=False)
    for batch_idx, (images, labels) in enumerate(dataloader):
        for k, image in enumerate(images[:12]):
            print('label:', labels[k])
            plt.subplot(2,6,k+1)
            plt.imshow(image.numpy().reshape(28,28), cmap='gray')
            plt.show()
        break























