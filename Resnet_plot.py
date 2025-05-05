import os
import csv
import matplotlib.pyplot as plt

# Define paths
data_dir = os.path.join("ResNET Results")
csv_path = os.path.join(data_dir, "pgd_evaluation.csv")
log_path = os.path.join(data_dir, "CIFAR100.log")

# Read CSV data for label flipping (Case 4 proportion)
epochs_csv = []
label_flip_prop = []
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            epoch = int(row['Epoch'])
            # Compute proportion of Case 4 over all misclassified examples
            case1 = float(row['Case1_pct'])
            case2 = float(row['Case2_pct'])
            case3 = float(row['Case3_pct'])
            case4 = float(row['Case4_pct'])
        except KeyError:
            continue
        total_mis = case1 + case2 + case3 + case4
        # Avoid division by zero
        if total_mis > 0:
            prop = case4 / total_mis
        else:
            prop = 0.0
        epochs_csv.append(epoch)
        label_flip_prop.append(prop * 100.0)  # convert to percentage

# Sort by epoch just in case
if epochs_csv:
    epochs_csv, label_flip_prop = zip(*sorted(zip(epochs_csv, label_flip_prop)))

# Read log file and extract accuracies
epochs_log = []
clean_acc = []
pgd_acc = []
fgsm_acc = []
with open(log_path, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[2] == '-':
            # Check if this line has an epoch number
            try:
                epoch = int(parts[3])
            except:
                continue
            # After epoch, values are from index 4 onward
            values = parts[4:]
            if len(values) >= 11:
                try:
                    # Column structure:
                    # [seconds, lr, inner_loss, train_loss, train_acc, test_loss, test_acc,
                    #  PGD_loss, PGD_acc, FGSM_loss, FGSM_acc, CW_loss, CW_acc]
                    test_acc_val = float(values[6])
                    pgd_acc_val = float(values[8])
                    fgsm_acc_val = float(values[10])
                except:
                    continue
                epochs_log.append(epoch)
                clean_acc.append(test_acc_val * 100.0)  # to percentage
                pgd_acc.append(pgd_acc_val * 100.0)
                fgsm_acc.append(fgsm_acc_val * 100.0)

# Sort log data by epoch
log_data = sorted(zip(epochs_log, clean_acc, pgd_acc, fgsm_acc))
if log_data:
    epochs_log, clean_acc, pgd_acc, fgsm_acc = zip(*log_data)
else:
    epochs_log = []
    clean_acc = []
    pgd_acc = []
    fgsm_acc = []

# Plotting
plt.figure(figsize=(8,6))
plt.plot(epochs_log, clean_acc, label='Test Clean Accuracy', color='blue', linewidth=2)
plt.plot(epochs_log, pgd_acc, label='Test PGD-10 Accuracy', color='green', linewidth=2)
plt.plot(epochs_log, fgsm_acc, label='Train FGSM Accuracy', color='orange', linewidth=2)
plt.plot(epochs_csv, label_flip_prop, label='Label Flipping Proportion', color='red', linewidth=2, linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Accuracy / Proportion (%)')
plt.title('Catastrophic Overfitting (CO) and Label Flipping (CIFAR-100)')
plt.legend()
plt.grid(True)

# Save the figure
output_path = os.path.join(data_dir, "recreated_figure1a.png")
plt.savefig(output_path)
plt.close()
