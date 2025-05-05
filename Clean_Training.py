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
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--save-epoch", default=111,type=int)
    parser.add_argument('--dataset', default='CIFAR100',choices=['CIFAR10', 'CIFAR100'], help='which dataset to train on')
    # ouput
    parser.add_argument('--out-dir', default='TDAT', type=str, help='Output directory')
    parser.add_argument('--log', default="output.log", type=str)
    return parser.parse_args()

def main():
    args = get_args()
    # Setup output directory and logging
    output_path = args.out_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, args.log)
    print('device:', device)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, args.log))
    logger.info(args)

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Data loaders
    if args.dataset == 'CIFAR10':
        train_loader, test_loader = cifar10_get_loaders(args.data_dir, args.batch_size)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        train_loader, test_loader = cifar100_get_loaders(args.data_dir, args.batch_size)
        num_classes = 100

    # Initialize model
    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18()
    elif args.model == "PreActResNest18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    elif args.model == "ResNet34":
        model = ResNet34()
    elif args.model == "DeiT-Small":
        model = DeiT_Small_P4_32(num_classes=num_classes)
        
    model=torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()

    # Optimizer and learning-rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    num_examples = 50000  # CIFAR-10/100 training set size
    batch_size = args.batch_size
    iter_per_epoch = num_examples // batch_size + (0 if num_examples % batch_size == 0 else 1)
    total_steps = args.epochs * iter_per_epoch
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=total_steps / 2, step_size_down=total_steps / 2
        )
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(total_steps * args.milestone1 / args.epochs),
                        int(total_steps * args.milestone2 / args.epochs)],
            gamma=0.1
        )
    else:
        scheduler = None

    # Training loop
    logger.info('Epoch \t Seconds \t \t LR \t \t Train Loss \t Train Acc \t \t Test Loss \t Test Acc')
    epoch_clean_list = []
    best_test_acc = 0.0

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Train for one epoch on clean data
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # Forward pass on clean inputs
            outputs = model(X)
            loss = F.cross_entropy(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            # Optional: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.max(1)[1] == y).sum().item()
            train_total += y.size(0)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr_max

        # Evaluate on test set (clean accuracy)
        # Create a separate model instance for evaluation (to use clean eval without affecting training model)
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
            model_test = DeiT_Small_P4_32(num_classes=num_classes).to(device)
        else:
            model_test = None

        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        test_loss, test_acc, _, _ = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)

        # Log epoch statistics
        logger.info('%d \t %.1f \t \t %.4f \t \t%.4f \t %.4f \t \t %.4f \t %.4f',
                    epoch, epoch_time, current_lr,
                    train_loss / train_total, train_correct / train_total,
                    test_loss, test_acc)

        # Save model checkpoint
        ckpt_name = f'{args.model}_{args.dataset}_Standard_clean_acc_{test_acc:.4f}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_test.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / train_total
        }, os.path.join(args.out_dir, ckpt_name))

    model_test = torch.nn.DataParallel(
    DeiT_Small_P4_32(num_classes=num_classes).to(device)
    )
    # Log the clean accuracy over epochs
    logger.info(epoch_clean_list)
    # If you saved the last epoch under a known name, load it; otherwise reuse current weights:
    # ckpt = torch.load(os.path.join(args.out_dir, ckpt_name))
    # model_test.load_state_dict(ckpt['model_state_dict'])
    model_test.load_state_dict(model.state_dict())
    model_test.eval()

    # Clean accuracy (redundant, but for consistency)
    test_loss, test_acc, clean_preds, labels = evaluate_standard(test_loader, model_test)
    logger.info("Final Clean  | Loss: %.4f  Acc: %.4f", test_loss, test_acc)

    # FGSM (1‐step) attack
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test, restarts=1)
    logger.info("FGSM (ε=8/255)  | Loss: %.4f  Acc: %.4f", fgsm_loss, fgsm_acc)

    # PGD‐10 (10 steps)
    pgd_loss, pgd_acc, _, _, _, _, _ = evaluate_pgd(test_loader, model_test,
                                                    clean_preds, labels,
                                                    attack_iters=10, restarts=1)
    logger.info("PGD-10 (ε=8/255) | Loss: %.4f  Acc: %.4f", pgd_loss, pgd_acc)

    # CW ℓ∞ attack (10 steps)
    cw_loss, cw_acc = evaluate_pgd_cw(test_loader, model_test,
                                      attack_iters=10, restarts=1)
    logger.info("CW (ε=8/255)    | Loss: %.4f  Acc: %.4f", cw_loss, cw_acc)

if __name__ == "__main__":
    main()
