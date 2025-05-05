# Evaluating and Extending Taxonomy Driven Fast Adversarial Training

Code for CS 680A: AI Security Final Project

## Contributors

- Sai Manikandan (gsai29)
  
Sai handled the model fine‑tuning and our Spiedie setup. He tuned every BERT variant we used and then fine‑tuned the DeiT vision models. He also took my single‑GPU training code, adapted it for Spiedie, and got the finetuned DeiT runs working smoothly across multiple GPUs. On top of that, he integrated the AutoAttack adversarial evaluation into our existing pipeline and wrote the main parts of the project report, making sure our methods and results are clear.

- Shyam Kannan (musical-shyam)
  
I built the DeiT pipelines from the ground up. I coded the AT and TDAT versions of the DeiT models from scratch, trained the full set of DeiTs, and added the taxonomy module that sorts each sample into its five cases. I wrote the scripts that turn our logs into the final graphs, created the presentation slides, and updated the GitHub docs so anyone who clones the repo can run everything end‑to‑end.

## Environments

- python 
- torch    
- torchvision
- numpy>=1.21.0
- Pillow>=8.0
- tqdm
- matplotlib 

## Files and Folders

### Running TDAT code

- Train ResNet18 on CIFAR-10 with TDAT.

`python TDAT.py --epochs 2 --batch-m 0.75 --delta-init "random" --out-dir "TDAT" --log "CIFAR10.log" --model "ResNet18" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1 --dataset CIFAR10`

- Train DeiT-Small on CIFAR-100 with TDAT.

`python TDAT.py --epochs 2 --batch-m 0.75 --delta-init "random" --out-dir "TDAT" --log "DeiT.log" --model "DeiT-Small" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1 --dataset CIFAR100`

### Running AT code

- Train ResNet18 on CIFAR-100 with AT.

`python AT.py --epochs 1 --delta-init "random" --out-dir "AT" --log "ResNet_AT.log" --model "ResNet18" --save-epoch 1 --dataset CIFAR100`

- Train Deit-Small on CIFAR-100 with AT.

`python AT.py --epochs 1 --delta-init "random" --out-dir "AT" --log "Deit_AT.log" --model "DeiT-Small" --save-epoch 1 --dataset CIFAR100`

### multi_gpu_files

### models

- This folder holds the codes for backbone architectures.

### DeiT_results/Resnet_results/DistilBert_results

- These folders store training logs, outputs and different graphs for the respective datasets.

### DistilBert_scripts

- This folder stores the archived training code for Distilbert model

### miscellaneous_scripts

- This folder stores the code for some of the experimentations and graphs plotted in the project

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
