import sys
import argparse
import matplotlib.pyplot as plt

def parse_training_log(log_path):
    """
    Parse the training log file to extract per-epoch metrics.
    Returns a dictionary with keys:
      'epoch', 'clean_acc', 'pgd_acc', 'case1', 'case2', 'case3', 'case4', 'case5'
    Each value is a list indexed by epoch.
    """
    data = {'epoch': [], 'clean_acc': [], 'pgd_acc': [], 
            'case1': [], 'case2': [], 'case3': [], 'case4': [], 'case5': []}
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip lines that contain non-data (headers or config)
            if line.startswith('['):
                # Remove leading timestamp and logger formatting
                # Find the closing bracket of timestamp and the dash after it
                end_idx = line.find('] - ')
                if end_idx != -1:
                    content = line[end_idx+3:].strip()  # after "] -"
                else:
                    content = line  # in case format is different
            else:
                content = line
            # Now 'content' should start with either an epoch number or something like "Epoch"
            if content.startswith('Epoch') or content.startswith('Namespace'):
                # This is a header or config line, skip it
                continue
            # At this point, content is expected to begin with the epoch number
            parts = content.split()
            # The expected format (19 columns: epoch + 18 metrics including 5 cases)
            if len(parts) >= 19:
                try:
                    epoch_num = int(parts[0])
                except ValueError:
                    continue  # if first part isn't an integer, skip (just a safe guard)
                # Parse relevant fields
                # parts indices (0-based): 0=epoch, 7=test_loss, 8=test_acc, 9=pgd_loss, 10=pgd_acc, ... 14=cw_acc, 15=case1, 16=case2, 17=case3, 18=case4, 19=case5
                # But if len(parts)==19, last index is 18 (0-18). Actually in our log format, it should be 19 entries (0 through 18).
                # Let's recalc indices: we have identified 19 columns including epoch.
                # Index for Test Acc = 8, PGD Acc = 10, Case1 = 15, Case2=16, Case3=17, Case4=18, Case5=19 (19th would be index 18 if total len 19).
                # It's possible the log line is split or missing some fields, so use try/except accordingly.
                try:
                    test_acc = float(parts[8])
                    pgd_acc = float(parts[10])
                    case1 = float(parts[15])
                    case2 = float(parts[16])
                    case3 = float(parts[17])
                    case4 = float(parts[18])
                    case5 = float(parts[19]) if len(parts) > 19 else float(parts[18])  # adjust if indexing off by one
                except IndexError:
                    # If the indexing didn't match, skip this line
                    continue
                # Store values
                data['epoch'].append(epoch_num)
                # Convert accuracy fractions to percentages for plotting (0.0813 -> 8.13%)
                data['clean_acc'].append(test_acc * 100.0)
                data['pgd_acc'].append(pgd_acc * 100.0)
                # Case values are already percentages in the log
                data['case1'].append(case1)
                data['case2'].append(case2)
                data['case3'].append(case3)
                data['case4'].append(case4)
                data['case5'].append(case5)
    return data

def plot_co_and_label_flipping(epoch_list, clean_acc, pgd_acc, case4, outfile):
    """
    Plot clean vs PGD-10 accuracy over epochs, with Case4 percentage on a secondary y-axis.
    Saves the plot to the given outfile path.
    """
    fig, ax1 = plt.subplots(figsize=(8,6))
    # Plot clean and PGD accuracies on primary y-axis
    ax1.plot(epoch_list, clean_acc, label='Clean Test Accuracy', color='C0', marker='o')
    ax1.plot(epoch_list, pgd_acc, label='PGD-10 Test Accuracy', color='C1', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    # Add secondary y-axis for Case4 percentage (label flipping)
    ax2 = ax1.twinx()  # create twin axis sharing x-axis
    ax2.plot(epoch_list, case4, label='Case 4 (%)', color='red', linestyle='--', marker='^')
    ax2.set_ylabel('Case 4 Percentage (%)', color='red')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='red')
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Catastrophic Overfitting and Label Flipping')
    fig.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)

def plot_taxonomy_distribution(epoch_list, cases_data, outfile):
    """
    Plot the percentage of each case (1-5) over epochs.
    cases_data: dict with keys 'case1'..'case5' containing percentage lists.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epoch_list, cases_data['case1'], label='Case 1', color='green')
    ax.plot(epoch_list, cases_data['case2'], label='Case 2', color='purple')
    ax.plot(epoch_list, cases_data['case3'], label='Case 3', color='blue')
    ax.plot(epoch_list, cases_data['case4'], label='Case 4', color='red')
    ax.plot(epoch_list, cases_data['case5'], label='Case 5', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage of Examples (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Taxonomy Distribution of Adversarial Examples (Cases 1-5)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse AT/TDAT training log and generate plots.")
    parser.add_argument('log_file', type=str, help="Path to the training log file (e.g., Deit_AT.log)")
    parser.add_argument('--out-dir', type=str, default='.', help="Directory to save output plots")
    args = parser.parse_args()
    log_path = args.log_file
    out_dir = args.out_dir.rstrip('/')
    # Parse the log file
    metrics = parse_training_log(log_path)
    if not metrics['epoch']:
        print("No epoch data found in log. Please check the log format.")
        sys.exit(1)
    epochs = metrics['epoch']
    # Generate Figure 1: CO and label flipping plot
    co_plot_path = f"{out_dir}/catastrophic_overfitting_label_flipping.png"
    plot_co_and_label_flipping(epochs, metrics['clean_acc'], metrics['pgd_acc'], metrics['case4'], co_plot_path)
    # Generate Figure 2: Taxonomy distribution plot
    tax_plot_path = f"{out_dir}/taxonomy_distribution.png"
    plot_taxonomy_distribution(epochs, 
                               {'case1': metrics['case1'], 'case2': metrics['case2'],
                                'case3': metrics['case3'], 'case4': metrics['case4'], 'case5': metrics['case5']},
                               tax_plot_path)
    print(f"Saved plots: {co_plot_path}, {tax_plot_path}")
