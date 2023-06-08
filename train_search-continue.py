# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from config_utils import load_config
from datasets import get_datasets, SearchDataset
from procedures import prepare_seed, prepare_logger
from procedures import Buffer_Reward_Generator
from log_utils import time_string
from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces
from torch.distributions import Categorical

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# 原本是3e-4，0.0003
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
# 0.003
# 0.07和0.0003

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    # print(model.cells[0]._ops[0]) # by zzz
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    baseline = ExponentialMovingAverage(0.9)  # xargs.EMA_momentum)
    search_space = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
                    'dil_conv_3x3', 'dil_conv_5x5']
    te_reward_generator = Buffer_Reward_Generator(search_space, train_queue, valid_queue, 10)
    accuracy_history = []  # for 201: save gt accuracy

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    '''
    self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [self.alphas_normal,self.alphas_reduce,]
    def arch_parameters(self): return self._arch_parameters
    0.07和0.0003
    '''
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info('alphas_normal = %s', str(F.softmax(model.alphas_normal, dim=-1).cpu().detach().numpy()))
        logging.info('alphas_reduce = %s', str(F.softmax(model.alphas_reduce, dim=-1).cpu().detach().numpy()))
        #print(F.softmax(model.alphas_normal, dim=-1))
        #print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,te_reward_generator,baseline)
        logging.info('train_acc %f', train_acc)

        # validation
        # valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, te_reward_generator,baseline):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    for step, (input, target) in enumerate(train_queue):
        print(model.alphas_normal)
        print(F.softmax(model.alphas_normal, dim=-1))
        model.train()
        n = input.size(0)
        
        # model.arch_parameters()是关于alphas的列表，每个元素是1*8的矩阵
        log_prob, action = select_action(architect.model.arch_parameters())
        arch = architect.generate_arch(action)  # CellStructure object for nas-bench-201, arch_params (Tensor) for DARTS
        reward = te_reward_generator.step(arch)
        baseline.update(reward)
        loss_wot = (-log_prob * (reward - baseline.value())).sum()

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,
                       loss_wot=loss_wot)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def select_action(arch_parameters):
    probs = [nn.functional.softmax(arch_parameters[0], dim=-1),
             nn.functional.softmax(arch_parameters[1], dim=-1)]

    m = [Categorical(prob) for prob in probs]
    # DARTS, -1 for not using an edge, mute some edges by index_of_action of prob
    actions = [_m.sample() for _m in m]
    for cell_idx, action in enumerate(actions):
        cum_edges = 2
        # start from the 2nd block
        for block_idx in range(1, 4):
            _logit = []
            for edge in range(2 + block_idx):
                _logit.append(arch_parameters[cell_idx][edge + cum_edges, actions[cell_idx][edge + cum_edges]].item())
            selected_edges = np.random.choice(np.arange(2 + block_idx) + cum_edges, size=2, replace=False,
                                              p=torch.nn.functional.softmax(torch.Tensor(_logit).cuda(),
                                                                            dim=0).detach().cpu().numpy())
            # mute some edges
            for edge in range(2 + block_idx):
                if (edge + cum_edges) not in selected_edges:
                    actions[cell_idx][edge + cum_edges] = -1
            cum_edges += 2 + block_idx
    return torch.cat(
        [torch.index_select(_m.log_prob(_action.clamp(0)), 0, torch.where(_action >= 0)[0]) for _m, _action in
         zip(m, actions)], dim=0), [action.cpu().tolist() for action in actions]


if __name__ == '__main__':
    main()
