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
import copy, csv

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
        X, y = X.to(device), y.to(device)
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            
            pred = output.argmax(dim=1)
            pgd_preds_list.append(pred)
            
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    pgd_preds = torch.cat(pgd_preds_list, dim=0)
    
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
                
    total_samples = len(true_labels)
    case1_pct = case1 / total_samples * 100
    case2_pct = case2 / total_samples * 100
    case3_pct = case3 / total_samples * 100
    case4_pct = case4 / total_samples * 100
    case5_pct = case5 / total_samples * 100
    return pgd_loss / n, pgd_acc / n, case1_pct, case2_pct, case3_pct, case4_pct, case5_pct


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
            
            pred =  output.argmax(dim=1)
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

def log_tdat_metrics(epoch, model, train_loader, test_loader, 
                     model_name, dataset_name, total_epochs, storage=[]):
    """
    Log taxonomy case counts and accuracies for the given epoch. 
    Append results to storage, and save to CSV when final epoch is reached.
    """
    device = next(model.parameters()).device  # use model's device (CPU/GPU)
    was_training = model.training
    model.eval()  # set model to eval for inference

    # Initialize counters for Cases 1-5
    case_counts = [0, 0, 0, 0, 0]  # indices 0->Case1, 1->Case2, ..., 4->Case5

    # Loop over training data to categorize each sample
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Clean prediction
        outputs_clean = model(inputs)
        pred_clean = outputs_clean.argmax(dim=1)
        # Adversarial example generation (e.g., single-step FGSM)
        inputs_adv = inputs.clone().detach().requires_grad_(True)
        # Compute loss and gradient for FGSM
        loss = F.cross_entropy(model(inputs_adv), labels)
        model.zero_grad()
        loss.backward()
        # FGSM step: perturb inputs in the direction of gradient sign
        epsilon = 8/255.0  # example perturbation size (adjust as needed)
        inputs_adv = inputs_adv + epsilon * inputs_adv.grad.sign()
        inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)  # keep in valid range
        inputs_adv = inputs_adv.detach()
        # Adversarial prediction
        outputs_adv = model(inputs_adv)
        pred_adv = outputs_adv.argmax(dim=1)
        # Tally cases for each sample in the batch
        for j in range(labels.size(0)):
            y_true = labels[j]
            pc = pred_clean[j]
            pa = pred_adv[j]
            if pc == y_true:
                if pa == y_true:
                    case_counts[1] += 1  # Case 2: clean correct, adv also correct
                else:
                    case_counts[0] += 1  # Case 1: clean correct, adv misclassified
            else:  # clean prediction is wrong
                if pa == y_true:
                    case_counts[3] += 1  # Case 4: clean wrong, adv flips to correct
                elif pa == pc:
                    case_counts[2] += 1  # Case 3: clean wrong, adv still predicts same wrong class
                else:
                    case_counts[4] += 1  # Case 5: clean wrong, adv misclassified to a different wrong class

    # Compute clean accuracy on test set
    clean_correct = 0
    total_test = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
    clean_acc = 100.0 * clean_correct / total_test

    # Compute robust accuracy on test set (e.g., PGD-10 attack for robustness)
    robust_correct = 0
    # Example PGD-10 attack loop
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Perform a 10-step PGD attack on inputs
        X_adv = inputs.clone().detach()
        epsilon = 8/255.0
        alpha = 2/255.0
        X_adv += 0  # just to ensure X_adv exists
        for _ in range(10):  # 10 PGD iterations
            X_adv.requires_grad_(True)
            loss = F.cross_entropy(model(X_adv), labels)
            model.zero_grad()
            loss.backward()
            # Gradient ascent step
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # Project perturbation onto epsilon-ball
            delta = torch.clamp(X_adv - inputs, min=-epsilon, max=epsilon)
            X_adv = torch.clamp(inputs + delta, 0.0, 1.0).detach()
        # Evaluate model on adversarial inputs
        preds_adv = model(X_adv).argmax(dim=1)
        robust_correct += (preds_adv == labels).sum().item()
    robust_acc = 100.0 * robust_correct / total_test

    # Store this epoch's results (epoch index + 1 for human-readable epoch number)
    storage.append([epoch + 1, *case_counts, clean_acc, robust_acc])

    # If this is the last epoch, save all logged data to CSV
    if epoch + 1 == total_epochs:  
        filename = f"TDAT_{model_name}_{dataset_name}_{total_epochs}epochs_results.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Epoch", "Case1", "Case2", "Case3", "Case4", "Case5", 
                             "CleanAccuracy", "RobustAccuracy"])
            # Write each epoch's data row
            writer.writerows(storage)
        print(f"[TDAT] Saved analysis data to {filename}")

    # Restore model training mode if it was in training mode
    if was_training:
        model.train()
