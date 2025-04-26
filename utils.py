import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.utils.data as data
import copy

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).to(device)
std = torch.tensor(cifar10_std).view(3, 1, 1).to(device)

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize(X):
    return (X - mu) / std


def cifar10_get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

# cifar100_mean = (0.5071, 0.4865, 0.4409)
# cifar100_std  = (0.2673, 0.2564, 0.2761)

cifar100_mean = (0.0, 0.0, 0.0)   
cifar100_std = (1.0, 1.0, 1.0)    

def cifar100_get_loaders(dir_, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    train_set = datasets.CIFAR100(dir_, train=True,
                                  transform=train_tf, download=True)
    test_set  = datasets.CIFAR100(dir_, train=False,
                                  transform=test_tf , download=True)
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=0, pin_memory=True)
    test_loader  = data.DataLoader(test_set , batch_size, shuffle=False,
                                   num_workers=0, pin_memory=True)
    return train_loader, test_loader

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


# def evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon=(8 / 255.) / std):
#     alpha = (2 / 255.) / std
#     pgd_loss = 0
#     pgd_acc = 0
#     n = 0
#     model.eval()
#     for i, (X, y) in enumerate(test_loader):
#         X, y = X.to(device), y.to(device)
#         pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
#         with torch.no_grad():
#             output = model(normalize(X + pgd_delta))
#             loss = F.cross_entropy(output, y)
#             pgd_loss += loss.item() * y.size(0)
#             pgd_acc += (output.max(1)[1] == y).sum().item()
#             n += y.size(0)
#     return pgd_loss / n, pgd_acc / n


def evaluate_pgd(test_loader, model, clean_preds, true_labels, attack_iters, restarts, epsilon=(8 / 255.) / std):
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0

    case1 = 0
    case2 = 0
    case3 = 0
    case4 = 0
    case5 = 0

    pgd_preds_list = []

    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            pred = output.argmax(dim=1)

            pgd_preds_list.append(pred)

            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (pred == y).sum().item()
            n += y.size(0)

    # After looping through all batches
    pgd_preds = torch.cat(pgd_preds_list, dim=0)

    # Figure out the 5 cases
    for i in range(len(true_labels)):
        clean_correct = (clean_preds[i] == true_labels[i]).item()
        pgd_correct = (pgd_preds[i] == true_labels[i]).item()

        if clean_correct and not pgd_correct:
            case1 += 1
        elif clean_correct and pgd_correct:
            case2 += 1
        elif not clean_correct:
            if pgd_preds[i] == clean_preds[i]:
                case3 += 1
            elif pgd_preds[i] == true_labels[i]:
                case4 += 1
            else:
                case5 += 1

    # Compute percentages
    total_samples = len(true_labels)
    case1_pct = case1 / total_samples * 100
    case2_pct = case2 / total_samples * 100
    case3_pct = case3 / total_samples * 100
    case4_pct = case4 / total_samples * 100
    case5_pct = case5 / total_samples * 100

    return (pgd_loss / n, pgd_acc / n, 
            case1_pct, case2_pct, case3_pct, case4_pct, case5_pct)


def attack_fgsm(model, X, y, epsilon, alpha, restarts):
    attack_iters = 1
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_fgsm(test_loader, model, restarts):
    epsilon = (8 / 255.) / std
    alpha = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        pgd_delta = attack_fgsm(model, X, y, epsilon, alpha, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


# def evaluate_standard(test_loader, model):
#     test_loss = 0
#     test_acc = 0
#     n = 0

#     model.eval()
#     with torch.no_grad():
#         for i, (X, y) in enumerate(test_loader):
#             X, y = X.to(device), y.to(device)
#             output = model(X)

#             #counting for cases
#             pred = output.argmax(dim=1)

#             loss = F.cross_entropy(output, y)
#             test_loss += loss.item() * y.size(0)
#             test_acc += (output.max(1)[1] == y).sum().item()
#             n += y.size(0)
#     return test_loss / n, test_acc / n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0

    all_clean_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)

            pred = output.argmax(dim=1)  # prediction
            all_clean_preds.append(pred)
            all_labels.append(y)

            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    clean_preds = torch.cat(all_clean_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return test_loss / n, test_acc / n, clean_preds, labels



def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd_cw(test_loader, model, attack_iters, restarts):
    alpha = (2 / 255.) / std
    epsilon = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        pgd_delta = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters=attack_iters, restarts=restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


import numpy as np
from torch.autograd import Variable


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.to(device), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out



from autoattack import AutoAttack

def evaluate_autoattack(test_loader, model, epsilon=8/255., norm='Linf', attacks_to_run=['apgd-ce', 'apgd-dlr',  'square']):
    model.eval()
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='custom', attacks_to_run=attacks_to_run)

    all_inputs = []
    all_labels = []
    for X, y in test_loader:
        all_inputs.append(X)
        all_labels.append(y)
    X_test = torch.cat(all_inputs, dim=0).cuda()
    y_test = torch.cat(all_labels, dim=0).cuda()

    with torch.no_grad():
        X_adv = adversary.run_standard_evaluation(X_test, y_test, bs=test_loader.batch_size, return_labels=False)

    output = model(X_adv)
    acc = (output.argmax(dim=1) == y_test).float().mean().item()
    loss = F.cross_entropy(output, y_test).item()

    return loss, acc

def evaluate_apgd_ce(test_loader, model, epsilon=8/255.):
    return evaluate_autoattack(test_loader, model, epsilon, attacks_to_run=['apgd-ce'])

def evaluate_apgd_dlr(test_loader, model, epsilon=8/255.):
    return evaluate_autoattack(test_loader, model, epsilon, attacks_to_run=['apgd-dlr'])

def evaluate_square(test_loader, model, epsilon=8/255.):
    return evaluate_autoattack(test_loader, model, epsilon, attacks_to_run=['square'])
