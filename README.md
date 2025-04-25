# Evaluating and Extending Taxonomy Driven Fast Adversarial Training

Code for CS 680A: AI Security Final Project

## Contributors
- Sai Manikandan
- Shyam Kannan (musical-shyam)

## Environments

- python 3.7.13 (changed)
- torch 1.13.1 (changed)
- torchvision 0.13 (changed)

## Files

### TDAT.py

- Train ResNet18 on CIFAR-10 with TDAT.

`python TDAT.py --batch-m 0.75 --delta-init "random" --out-dir "TDAT" --log "CIFAR10.log" --model "ResNet18" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1 --dataset CIFAR10`

- Train DeiT on CIFAR-100 with TDAT.

`python TDAT.py --batch-m 0.75 --delta-init "random" --out-dir "CIFAR100" --log "DeiT.log" --model "DeiT-Small" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1 --dataset CIFAR100`

### models

- This folder holds the codes for backbones.

### CIFAR10/CIFAR100/TinyImageNet/ImageNet100

- These folders store training logs and outputs for the respective datasets.

## Trained Models
[Checkpoint on CIFAR-10 with our method](https://drive.google.com/file/d/1fPYwjz2V9wibfdWlopip0tfK4IB0KS9o/view?usp=drive_link)

## Citation
If you are insterested in this work, please consider citing:

```bibtex
@inproceedings{tong2024taxonomy,
  title={Taxonomy driven fast adversarial training},
  author={Tong, Kun and Jiang, Chengze and Gui, Jie and Cao, Yuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5233--5242},
  year={2024}
}
