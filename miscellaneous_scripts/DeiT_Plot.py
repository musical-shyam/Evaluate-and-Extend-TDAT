import matplotlib.pyplot as plt

def parse_log(filepath):
    """Parse the training log to extract epoch, Test Acc, PGD Acc, FGSM Acc, and Case1-5 percentages."""
    data = {'epoch':[], 'test_acc':[], 'pgd_acc':[], 'fgsm_acc':[],
            'case1':[], 'case2':[], 'case3':[], 'case4':[], 'case5':[]}
    with open(filepath, 'r') as f:
        for line in f:
            # Identify lines with epoch data (prefix '[... - ') and numeric epoch
            if '] - ' in line:
                parts = line.split('] - ', 1)
                if len(parts) < 2:
                    continue
                after = parts[1]
                tokens = [t.strip() for t in after.split('\t') if t.strip()]
                if not tokens:
                    continue
                # Check if first token is a digit (epoch number)
                if tokens[0].isdigit():
                    epoch = int(tokens[0])
                    # Ensure we have all expected columns
                    if len(tokens) >= 17:
                        data['epoch'].append(epoch)
                        data['test_acc'].append(float(tokens[7]))
                        data['pgd_acc'].append(float(tokens[9]))
                        data['fgsm_acc'].append(float(tokens[16]))
                        data['case1'].append(float(tokens[10]))
                        data['case2'].append(float(tokens[11]))
                        data['case3'].append(float(tokens[12]))
                        data['case4'].append(float(tokens[13]))
                        data['case5'].append(float(tokens[14]))
    return data

# Process each log and create the plots
for logname in ['DeiT-AT.log', 'DeiT-TDAT-Finetuning.log']:
    data = parse_log(logname)
    epochs = data['epoch']
    
    # Compute label flipping proportion (%) = Case4 / (Case3+Case4+Case5)
    label_flip_prop = []
    for c3, c4, c5 in zip(data['case3'], data['case4'], data['case5']):
        total_mis = c3 + c4 + c5
        if total_mis > 0:
            label_flip_prop.append(c4 / total_mis * 100.0)
        else:
            label_flip_prop.append(0.0)
    
    # === Figure 1: CO and Label Flipping ===
    # plt.figure()
    # plt.plot(epochs, data['test_acc'], color='blue', label='Test Clean Accuracy')
    # plt.plot(epochs, data['pgd_acc'], color='green', label='Test PGD-10 Accuracy')
    # plt.plot(epochs, data['fgsm_acc'], color='orange', label='Train FGSM Accuracy')
    # plt.plot(epochs, label_flip_prop, 'r--', label='Label Flipping Proportion')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy / Proportion (%)')
    # plt.title('Catastrophic Overfitting (CO) and Label Flipping')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'figure1_like_{logname[:-4]}.png')
    # plt.close()

    # # === Figure 3: Taxonomy Cases (1-5) vs Epoch ===
    # plt.figure()
    # plt.plot(epochs, data['case1'], '-o', color='blue', label='Case 1')
    # plt.plot(epochs, data['case2'], '-o', color='orange', label='Case 2')
    # plt.plot(epochs, data['case3'], '-o', color='green', label='Case 3')
    # plt.plot(epochs, data['case4'], '-o', color='red', label='Case 4')
    # plt.plot(epochs, data['case5'], '-o', color='purple', label='Case 5')
    # plt.xlabel('Epoch')
    # plt.ylabel('Percentage of Test Set')
    # plt.title('Taxonomy Cases (1–5) vs Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'figure3_like_{logname[:-4]}.png')
    # plt.close()
    # -- AFTER you have parsed the log and built the full lists --

    # Keep only epochs that are multiples of 3
    keep_idx = [i for i, ep in enumerate(epochs) if ep % 3 == 0]

    # Slice every series with the same indices
    epochs_3   = [epochs[i]   for i in keep_idx]
    case1_3    = [data['case1'][i] for i in keep_idx]
    case2_3    = [data['case2'][i] for i in keep_idx]
    case3_3    = [data['case3'][i] for i in keep_idx]
    case4_3    = [data['case4'][i] for i in keep_idx]
    case5_3    = [data['case5'][i] for i in keep_idx]

    # ---------- Figure 3 (every‑3‑epochs version) ----------
    plt.figure()
    plt.plot(epochs_3, case1_3, '-o', color='blue',   label='Case 1')
    plt.plot(epochs_3, case2_3, '-o', color='orange', label='Case 2')
    plt.plot(epochs_3, case3_3, '-o', color='green',  label='Case 3')
    plt.plot(epochs_3, case4_3, '-o', color='red',    label='Case 4')
    plt.plot(epochs_3, case5_3, '-o', color='purple', label='Case 5')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of Test Set')
    plt.title('Taxonomy Cases (1–5) vs Epoch (every 3 epochs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figure3_like_{logname[:-4]}_every3.png')
    plt.close()

