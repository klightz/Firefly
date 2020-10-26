import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import Config
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Classifier
import sp
from torch.autograd import Variable
import math

def adjust_learning_rate(optimizer, epoch, total_epochs, lr):
    lr_type = 'cos'
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * lr * (1 + math.cos(math.pi * epoch / total_epochs))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

def test(testloader, model):
    model.eval()
    correct = 0
    total = 1e-40
    for x, y in testloader:
        x = x.cuda(); y
        y_ = model.classify(x).detach().cpu()
        correct += (y == y_).float().sum().item()
        total += x.shape[0]
    model.train()
    return correct / total

def run(trainset, trainloader, testloader, config):
    check_path('./img')

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    exp_name = "experiment_%s_%s_initdim%d_seed%d_grow%f_gra%d_alpha3_new" % (
            config.dataset,
            config.method,
            config.dim_hidden,
            config.seed,
            config.grow_ratio,
            config.granularity)

    config.resume = False
    if config.resume:
        load_round = config.load_round
        stats = np.load("stats/round_%d_%s.npy" % (load_round, exp_name), allow_pickle=True)
        stats = stats.tolist()
        model = Classifier(config, stats['cfg']).to(config.device)
        ckpt = torch.load("stats/round_%d_%s.pt" % (load_round, exp_name))
        model.load_state_dict(ckpt)

    else:
        model = Classifier(config).to(config.device)
        if config.verbose:
            print('[INFO] initial model trainable parameter number: %d' % model.get_num_params())


        stats = {
            'train_loss': [],
            'test_accuracy': [],
            'compression_rate': [],
            'widths': {},
            'cfg': None,
        }

        for i, layer in enumerate(model.net):
            if isinstance(layer, sp.SpModule) and layer.can_split:
                if config.method == 'fireflyn':
                    stats['widths'][i] = [0, 0]
                else:
                    stats['widths'][i] = 0

    n_batches = len(trainloader)
    print("[INFO] Split method: ", config.method)

    if config.resume:
        load_round = config.load_round
        ckpt = torch.load("checkpoint/roundfull_%d_%s.pt" % (load_round, exp_name))
        model.load_state_dict(ckpt)
        print('load succ')
    
    load_round = config.load_round
    for round in range(load_round, 10 + 1):
        for epoch in range(1, config.n_epochs+1):
            if round <= load_round and config.resume:
                break
            loss = 0.
            for i, (x, y) in enumerate(trainloader):
                inputs = x.to(config.device); targets = y.to(config.device)
                loss += model.update(inputs, targets)
            loss /= n_batches
            test_acc = test(testloader, model)
            stats['train_loss'].append(loss)
            stats['test_accuracy'].append(test_acc)

            if epoch % 5 == 0:
                print("[INFO] Round %d Epoch %05d | Training loss is %10.4f | Test accuracy is %10.4f" % (round, epoch, loss, test_acc))

            if epoch == config.n_epochs // 2 - 1:
                model.decay_lr(0.1)

            if epoch == config.n_epochs // 4 * 3 - 1:
                model.decay_lr(0.1)

            if epoch % 20 == 0 or epoch == config.n_epochs:
                np.save("checkpoint/%s.npy" % exp_name, stats)

            if epoch % 20 == 0 or epoch == config.n_epochs:
                torch.save(model.state_dict(), "checkpoint/%s.pt" % exp_name)

        np.save("checkpoint/roundfull_%d_%s.npy" % (round, exp_name), stats)
        torch.save(model.state_dict(), "checkpoint/roundfull_%d_%s.pt" % (round, exp_name))

        if config.method != 'none':
            # Grow the network use NASH
            if config.method == 'random':
                best_acc = 0
                rtime = time.time()
                for n in range(8):
                    newmodel = copy.deepcopy(model)
                    newmodel.set_lr(0.05)
                    newmodel.create_optimizer()
                    n_neurons = newmodel.split(config.method, trainset)
                    loss = 0.
                    for e in range(17):
                        print(e)
                        for i, (x, y) in enumerate(trainloader):
                            inputs = x.to(config.device); targets = y.to(config.device)
                            loss += newmodel.update(inputs, targets)
                        adjust_learning_rate(newmodel.opt, e, 17, 0.05)
                        test_acc = test(testloader, newmodel)
                    if test_acc > best_acc:
                        bestmodel = copy.deepcopy(newmodel)
                    del newmodel
                model = copy.deepcopy(bestmodel)
                del bestmodel
                print('Search Time', time.time() - rtime)
            else:
                n_neurons = model.split(config.method, trainset)

            print('Current parameter size' , model.get_num_params())
            CFG = model.get_cfg()
            print('Current cfg', model.get_cfg())
            model.set_lr(0.1)
            lr = 0.1
            model.create_optimizer()

if __name__ == "__main__":
    config = Config()
    if not os.path.exists('stats'):
        os.makedirs('stats')
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), [0.2470, 0.2435, 0.2616]),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), [0.2470, 0.2435, 0.2616]),
    ])
    if config.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='../../ButterFly/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.MNIST(root='../../ButterFly/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    elif config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../../ButterFly/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='../../ButterFly/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    elif config.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='../../ButterFly/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR100(root='../../ButterFly/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    run(trainset, trainloader, testloader, config)
