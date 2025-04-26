
import argparse
import logging
import os
import torch
from models import *
from utils import *
import numpy as np

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='Model architecture')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--log', default="TDAT_test_eval.log", type=str)
    return parser.parse_args()

def main():
    args = get_args()

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=args.log)
    logger.info(args)

    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    _, test_loader = cifar100_get_loaders(args.data_dir, args.batch_size)

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
    else:
        raise ValueError("Invalid model selected")

    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    logger.info("Evaluating...")

    test_loss, test_acc = evaluate_standard(test_loader, model)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model, restarts=1)
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
    cw_loss, cw_acc = evaluate_pgd_cw(test_loader, model, attack_iters=10, restarts=1)



    autoattack_loss, autoattack_acc = evaluate_autoattack(test_loader, model)
    apgd_ce_loss, apgd_ce_acc = evaluate_apgd_ce(test_loader, model)
    square_loss, square_acc = evaluate_square(test_loader, model)

    logger.info("Clean Eval: Loss = %.4f, Acc = %.4f", test_loss, test_acc)
    logger.info("FGSM Eval: Loss = %.4f, Acc = %.4f", fgsm_loss, fgsm_acc)
    logger.info("PGD Eval : Loss = %.4f, Acc = %.4f", pgd_loss, pgd_acc)
    logger.info("CW Eval  : Loss = %.4f, Acc = %.4f", cw_loss, cw_acc)


    logger.info("AutoAttack (full): Loss = %.4f, Acc = %.4f", autoattack_loss, autoattack_acc)
    logger.info("APGD-CE         : Loss = %.4f, Acc = %.4f", apgd_ce_loss, apgd_ce_acc)
    logger.info("Square Attack   : Loss = %.4f, Acc = %.4f", square_loss, square_acc)


if __name__ == "__main__":
    main()
