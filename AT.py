import argparse
import copy
import logging
import os
import time
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from torch.nn import functional as F
from models import *
import random
from torch.autograd import Variable
from utils import *
import math

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--milestone1', default=100, type=int)
    parser.add_argument('--milestone2', default=105, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--save-epoch", default=111,type=int)
    parser.add_argument('--dataset', default='CIFAR100',choices=['CIFAR10', 'CIFAR100'], help='which dataset to train on')
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'], help='Perturbation initialization method')
    # ouput
    parser.add_argument('--out-dir', default='AT', type=str, help='Output directory')
    parser.add_argument('--log', default="output.log", type=str)
    return parser.parse_args()



def label_relaxation(label, factor, num_classes):
    one_hot = np.eye(num_classes)[label.to(device).data.cpu().numpy()] 
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return result


def main():
    args = get_args()

    output_path = args.out_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, args.log)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, args.log))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'CIFAR10':
        train_loader, test_loader = cifar10_get_loaders(args.data_dir, args.batch_size)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        train_loader, test_loader = cifar100_get_loaders(args.data_dir, args.batch_size)
        num_classes = 100
        
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18(num_classes=num_classes)
    elif args.model == "PreActResNest18":
        model = PreActResNet18(num_classes=num_classes)
    elif args.model == "WideResNet":
        model = WideResNet(num_classes=num_classes)
    elif args.model == "ResNet34":
        model = ResNet34(num_classes=num_classes)
    elif args.model == "DeiT-Small":
        model = DeiT_Small_P4_32(num_classes=num_classes)
        
    model=torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_of_example = 50000
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)

    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * args.milestone1/args.epochs, lr_steps * args.milestone2/args.epochs],
                                                         gamma=0.1)

    # Training
    logger.info('Epoch \t Seconds \t LR \t Inner Loss \t Train Loss \t Train Acc \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        inner_loss = train_loss = train_acc = train_n = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # ---- FGSM (single-step) adversarial example -------------------
            model.eval()                                   # gradients off
            delta = attack_fgsm(model, X, y, epsilon, alpha,
                                restarts=1)                # 1-step FGSM
            model.train()

            # optional “inner” loss on clean inputs just for logging
            with torch.no_grad():
                ori_logits = model(X)
                ori_loss   = F.cross_entropy(ori_logits, y)
            inner_loss += ori_loss.item() * y.size(0)

            # ---- update on adversarial batch -----------------------------
            logits = model(X + delta)
            loss   = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            scheduler.step()

            train_loss += loss.item() * y.size(0)
            train_acc  += (logits.max(1)[1] == y).sum().item()
            train_n    += y.size(0)

        # Evaluation starts, same as TDAT
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        
        if args.model == "VGG":
            model_test = VGG('VGG19').to(device)
        elif args.model == "ResNet18":
            model_test = ResNet18().to(device)
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18().to(device)
        elif args.model == "WideResNet":
            model_test = WideResNet().to(device)
        elif args.model == "ResNet34":
            model_test = ResNet34().to(device)
        elif args.model == "DeiT-Small":
            model_test = DeiT_Small_P4_32().to(device)
            
        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)

        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr,inner_loss/train_n, train_loss / train_n, train_acc / train_n, test_loss, test_acc, pgd_loss, pgd_acc)
        # save checkpoints
        ckpt_name = args.model + "_" + args.dataset + "_TDAT_robustAcc_" + str(pgd_acc) + "_clean_acc_" + str(test_acc) + ".pt"  
        if epoch >= args.save_epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_test.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': train_loss/train_n
            },os.path.join(args.out_dir, ckpt_name))
        log_tdat_metrics(epoch, model, train_loader, test_loader, model_name="DeiTS", dataset_name="CIFAR100", total_epochs=args.save_epoch)
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)


if __name__ == "__main__":
    main()