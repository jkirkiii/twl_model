import re
import matplotlib.pyplot as plt


def parse_training_log(log_file_path):
    """
    Parse training log to extract validation losses for each fold
    """
    fold_data = {}
    current_fold = None
    debug_info = {'total_lines': 0, 'fold_lines': 0, 'epoch_lines': 0, 'parsed_epochs': 0}

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return fold_data, debug_info

    debug_info['total_lines'] = len(lines)

    for line in lines:
        line = line.strip()

        # Detect new fold
        if "Starting Fold" in line:
            debug_info['fold_lines'] += 1
            fold_match = re.search(r"Starting Fold (\d+)/\d+", line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                fold_data[current_fold] = {'epochs': [], 'val_losses': [], 'train_losses': []}
                print(f"Found fold {current_fold} in {log_file_path}")

        # Extract epoch data
        elif "Epoch [" in line and "Val Loss:" in line and current_fold is not None:
            debug_info['epoch_lines'] += 1

            epoch_match = re.search(r"Epoch \[(\d+)/\d+\]", line)
            val_loss_match = re.search(r"Val Loss: ([\d.e-]+)", line)
            train_loss_match = re.search(r"Train Loss: ([\d.e-]+)", line)

            if epoch_match and val_loss_match and train_loss_match:
                try:
                    epoch = int(epoch_match.group(1))
                    val_loss = float(val_loss_match.group(1))
                    train_loss = float(train_loss_match.group(1))

                    fold_data[current_fold]['epochs'].append(epoch)
                    fold_data[current_fold]['val_losses'].append(val_loss)
                    fold_data[current_fold]['train_losses'].append(train_loss)
                    debug_info['parsed_epochs'] += 1

                except ValueError as e:
                    print(f"Error parsing numbers in line: {line}")
                    print(f"Error: {e}")
            else:
                print(f"Failed to match epoch line: {line[:100]}...")

    print(f"Debug info for {log_file_path}: {debug_info}")
    print(f"Folds found: {list(fold_data.keys())}")
    for fold, data in fold_data.items():
        print(f"  Fold {fold}: {len(data['epochs'])} epochs")

    return fold_data, debug_info


def create_learning_curves_comparison():
    """
    Create a three-panel comparison of learning curves
    """
    log_files = {
        'FCNN': 'fcnn_train_log.txt',
        'ResNet': 'resnet_train_log.txt',
        'Spatial-Aware': 'spatial_train_log.txt'
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training and Validation Loss Convergence Across Architectures', fontsize=16, y=0.98)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (model_name, log_path) in enumerate(log_files.items()):
        try:
            fold_data, debug_info = parse_training_log(log_path)

            # Plot validation losses
            ax_val = axes[0, idx]
            # Plot training losses
            ax_train = axes[1, idx]

            if fold_data:
                # Plot each fold
                for fold_num in sorted(fold_data.keys()):
                    epochs = fold_data[fold_num]['epochs']
                    val_losses = fold_data[fold_num]['val_losses']
                    train_losses = fold_data[fold_num]['train_losses']

                    # Validation losses
                    ax_val.plot(epochs, val_losses,
                                color=colors[fold_num - 1],
                                label=f'Fold {fold_num}',
                                alpha=0.8, linewidth=1.5)

                    # Training losses
                    ax_train.plot(epochs, train_losses,
                                  color=colors[fold_num - 1],
                                  label=f'Fold {fold_num}',
                                  alpha=0.8, linewidth=1.5)

                # Formatting for validation loss plot
                ax_val.set_ylabel('Validation Loss')
                ax_val.set_title(f'{model_name} - Validation Loss')
                ax_val.grid(True, alpha=0.3)
                ax_val.legend(fontsize=8)

                # Formatting for training loss plot
                ax_train.set_xlabel('Epoch')
                ax_train.set_ylabel('Training Loss')
                ax_train.set_title(f'{model_name} - Training Loss')
                ax_train.grid(True, alpha=0.3)
                ax_train.legend(fontsize=8)

                # Set y-axis limits
                all_val_losses = []
                all_train_losses = []
                for fold in fold_data.values():
                    all_val_losses.extend(fold['val_losses'])
                    all_train_losses.extend(fold['train_losses'])

                if all_val_losses:
                    max_val_loss = max(all_val_losses)
                    min_val_loss = min(all_val_losses)
                    range_val = max_val_loss - min_val_loss
                    ax_val.set_ylim(max(0, min_val_loss - range_val * 0.05),
                                    max_val_loss + range_val * 0.2)

                if all_train_losses:
                    max_train_loss = max(all_train_losses)
                    min_train_loss = min(all_train_losses)
                    range_train = max_train_loss - min_train_loss
                    ax_train.set_ylim(max(0, min_train_loss - range_train * 0.05),
                                      max_train_loss + range_train * 0.2)

            else:
                ax_val.text(0.5, 0.5, f'No data found in:\n{log_path}',
                            ha='center', va='center', transform=ax_val.transAxes)
                ax_train.text(0.5, 0.5, f'No data found in:\n{log_path}',
                              ha='center', va='center', transform=ax_train.transAxes)

        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            axes[0, idx].text(0.5, 0.5, f'Error processing:\n{log_path}\n{str(e)}',
                              ha='center', va='center', transform=axes[0, idx].transAxes)
            axes[1, idx].text(0.5, 0.5, f'Error processing:\n{log_path}\n{str(e)}',
                              ha='center', va='center', transform=axes[1, idx].transAxes)

    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    create_learning_curves_comparison()
