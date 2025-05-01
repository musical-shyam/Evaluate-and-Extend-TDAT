import os, re, csv
import torch
import matplotlib.pyplot as plt
from models import *
from utils import *

import time
from datetime import datetime

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for log and checkpoints
log_file = 'CIFAR100.log'
ckpt_dir = 'TDAT_CIFAR100'

# 1. Parse the training log to list epochs
epochs_in_log = []
with open(log_file, 'r') as f:
    for line in f:
        # Match lines like "[... - 3 ..." to extract the epoch number
        match = re.search(r'\-\s+(\d+)\s+', line)
        if match:
            epochs_in_log.append(int(match.group(1)))
epochs_in_log = sorted(set(epochs_in_log))
print(f"Epochs in log: {epochs_in_log}")

# 2. Identify checkpoint files and their epochs
ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
epoch_to_file = {}
for fname in ckpt_files:
    data = torch.load(os.path.join(ckpt_dir, fname), map_location=device)
    if 'epoch' in data:
        epoch_to_file[int(data['epoch'])] = fname
sorted_epochs = sorted(epoch_to_file.keys())
print(f"Checkpoint epochs: {sorted_epochs}")

# Select every 3rd epoch (e.g. 3, 6, 9, ...)
selected_epochs = [ep for ep in sorted_epochs if ep % 3 == 0 and ep != 0]
print(f"Selected epochs (every 3rd): {selected_epochs}")

# 3. Prepare test data loader for CIFAR-100
_, test_loader = cifar100_get_loaders(dir_='./data', batch_size=128)

# Containers for results
robust_acc = []
case1_pct = []
case2_pct = []
case3_pct = []
case4_pct = []
case5_pct = []

# Open CSV log file
with open('pgd_evaluation.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Eval_Time_s', 'Epoch', 'PGD_Acc', 'Case1_pct', 'Case2_pct', 'Case3_pct', 'Case4_pct', 'Case5_pct'])
    # Evaluate each selected checkpoint
    for ep in selected_epochs:
        fname = epoch_to_file[ep]
        ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location=device)
        
        # Instantiate ResNet-18 for 100 classes
        model = ResNet18(num_classes=100).to(device)
        model.eval()
        
        # Load weights (strip DataParallel "module." prefix if present)
        state_dict = ckpt['model_state_dict']
        new_state = {}
        for key, val in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state[new_key] = val
        model.load_state_dict(new_state)
        
        start = time.time()
        # Evaluate on clean and adversarial test data
        _, clean_acc, clean_preds, true_labels = evaluate_standard(test_loader, model)
        pgd_loss, pgd_accuracy, c1, c2, c3, c4, c5 = evaluate_pgd(
            test_loader, model, clean_preds, true_labels, attack_iters=10, restarts=1)
        elapsed = time.time() - start
        now_ts = datetime.now().isoformat(timespec='seconds')
        # Store results (convert to percentages)
        robust_acc.append(pgd_accuracy * 100)
        case1_pct.append(c1)
        case2_pct.append(c2)
        case3_pct.append(c3)
        case4_pct.append(c4)
        case5_pct.append(c5)
        
        # Log to CSV
        writer.writerow([now_ts, f"{elapsed:.2f}",ep, pgd_accuracy * 100, c1, c2, c3, c4, c5])

# 4. Plot robust accuracy vs. epoch
plt.figure(figsize=(6,4))
plt.plot(selected_epochs, robust_acc, marker='o', color='C0')
plt.xlabel('Epoch')
plt.ylabel('Robust Accuracy (%)')
plt.title('PGD-10 Robust Accuracy vs Epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig('robust_accuracy_vs_epoch.png')

# 5. Plot taxonomy case percentages vs. epoch (cases 1–5)
plt.figure(figsize=(6,4))
plt.plot(selected_epochs, case1_pct, marker='o', label='Case 1')
plt.plot(selected_epochs, case2_pct, marker='o', label='Case 2')
plt.plot(selected_epochs, case3_pct, marker='o', label='Case 3')
plt.plot(selected_epochs, case4_pct, marker='o', label='Case 4')
plt.plot(selected_epochs, case5_pct, marker='o', label='Case 5')
plt.xlabel('Epoch')
plt.ylabel('Percentage of Test Set')
plt.title('Taxonomy Cases (1–5) vs Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('taxonomy_cases_vs_epoch.png')
