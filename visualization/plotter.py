import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_handling.create_Blured_data import BlurDataset

def show_blur_samples(test_data, save_path='outputs/blur_samples.png'):
    """
    Visualizes one sample image at Clean, Mild, and Heavy blur levels.
    Saves the image to `save_path`.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # CIFAR-10 normalization values
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    blur_names = ['Clean', 'Mild Blur', 'Heavy Blur']
    for level, ax, title in zip([0, 1, 2], axes, blur_names):
        ds = BlurDataset(test_data, blur_level=level)
        img = ds[0][0]  # First sample image (Tensor: 3, 224, 224)
        
        # Un-normalize back to viewable RGB: image * std + mean
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle("Sample Image at Different Blur Levels", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved sample images to: {save_path}")


def plot_comparison_graphs(blur_levels, std_accs, mt_accs, output_dir='outputs'):
    """
    Generates and saves:
      1. Line graph: Accuracy vs Blur Level
      2. Bar chart: Standard vs Multi-Task Comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Graph 1: Line graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(blur_levels, std_accs, 'o--', color='tomato', label='Standard MobileNetV2', linewidth=2)
    plt.plot(blur_levels, mt_accs, 's-', color='steelblue', label='Multi-Task MobileNetV2 (Ours)', linewidth=2)
    plt.title('Classification Accuracy vs Blur Level', fontsize=14)
    plt.xlabel('Blur Level')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    line_path = os.path.join(output_dir, 'line_graph.png')
    plt.savefig(line_path, dpi=150)
    plt.close()
    print(f"[OK] Saved line graph to: {line_path}")

    # --- Graph 2: Bar chart ---
    x = np.arange(len(blur_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, std_accs, width, label='Standard', color='tomato', alpha=0.85)
    bars2 = ax.bar(x + width/2, mt_accs, width, label='Multi-Task', color='steelblue', alpha=0.85)

    ax.set_xlabel('Blur Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Standard vs Multi-Task Model Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(blur_levels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f%%', padding=3)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    
    bar_path = os.path.join(output_dir, 'bar_chart.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[OK] Saved bar chart to: {bar_path}")


def print_summary_table(blur_levels, std_accs, mt_accs):
    """
    Prints a comparison table in the terminal.
    """
    print("\n" + "="*52)
    print(f"{'Blur Level':20} {'Standard':>12} {'Multi-Task':>12}")
    print("="*52)
    for level, s, m in zip(blur_levels, std_accs, mt_accs):
        diff = m - s
        sign = "+" if diff >= 0 else ""
        print(f"{level:20} {s:>11.2f}% {m:>11.2f}%   ({sign}{diff:.2f}%)")
    print("="*52)

