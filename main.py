import os
import torch
from torch.utils.data import DataLoader

# 1. Data Handling imports
from data_handling.downaload_data import get_data
from data_handling.create_Blured_data import create_blur_datasets

# 2. Plain Model imports
from plain_training.model import StandardMobileNetV2
from plain_training.train import train_standard, evaluate as evaluate_standard

# 3. Multi-Head Model imports
from multi_head_training.model import MultiTaskMobileNetV2
from multi_head_training.train import train_multitask, evaluate as evaluate_multitask

# 4. Visualization imports
from visualization.plotter import show_blur_samples, plot_comparison_graphs, print_summary_table


def main():
    # Setup Device & Output Directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output files and images will be saved in: '{OUTPUT_DIR}/'")

    # ============================================================
    # STEP 1: LOAD & PREPARE DATA
    # ============================================================
    print("\n[Step 1/4] Loading datasets...")
    train_raw, test_raw = get_data()

    # Save visual sample of blur levels
    show_blur_samples(test_raw, save_path=os.path.join(OUTPUT_DIR, 'blur_samples.png'))

    # Create blurred datasets
    datasets = create_blur_datasets(train_raw, test_raw)
    
    BATCH_SIZE = 64
    # Note: On Windows, num_workers=0 ensures stable multiprocessing
    loader_train_mixed = DataLoader(datasets["train_mixed"], batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    loader_test_clean  = DataLoader(datasets["test_clean"],  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loader_test_mild   = DataLoader(datasets["test_mild"],   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loader_test_heavy  = DataLoader(datasets["test_heavy"],  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ============================================================
    # STEP 2: TRAIN & EVALUATE STANDARD MODEL
    # ============================================================
    print("\n[Step 2/4] Training Standard MobileNetV2...")
    standard_model = StandardMobileNetV2(num_classes=10)
    standard_model = train_standard(standard_model, loader_train_mixed, device=device, epochs=5)

    print("Evaluating Standard Model...")
    std_acc_clean = evaluate_standard(standard_model, loader_test_clean, device=device, desc="Std Clean")
    std_acc_mild  = evaluate_standard(standard_model, loader_test_mild,  device=device, desc="Std Mild")
    std_acc_heavy = evaluate_standard(standard_model, loader_test_heavy, device=device, desc="Std Heavy")

    # Save Standard Model weights inside outputs/
    std_model_path = os.path.join(OUTPUT_DIR, 'standard_model.pth')
    torch.save(standard_model.state_dict(), std_model_path)
    print(f"[SAVED] Standard model weights to: {std_model_path}")

    # ============================================================
    # STEP 3: TRAIN & EVALUATE MULTI-TASK MODEL
    # ============================================================
    print("\n[Step 3/4] Training Multi-Task MobileNetV2...")
    multitask_model = MultiTaskMobileNetV2(num_classes=10, num_blur_levels=3)
    multitask_model = train_multitask(multitask_model, loader_train_mixed, device=device, epochs=5, blur_weight=0.3)

    print("Evaluating Multi-Task Model...")
    mt_acc_clean = evaluate_multitask(multitask_model, loader_test_clean, device=device, desc="MT Clean")
    mt_acc_mild  = evaluate_multitask(multitask_model, loader_test_mild,  device=device, desc="MT Mild")
    mt_acc_heavy = evaluate_multitask(multitask_model, loader_test_heavy, device=device, desc="MT Heavy")

    # Save Multi-Task Model weights inside outputs/
    mt_model_path = os.path.join(OUTPUT_DIR, 'multitask_model.pth')
    torch.save(multitask_model.state_dict(), mt_model_path)
    print(f"[SAVED] Multi-task model weights to: {mt_model_path}")

    # ============================================================
    # STEP 4: VISUALIZATION & COMPARISON
    # ============================================================
    print("\n[Step 4/4] Generating comparison plots & summary table...")
    blur_levels = ['Clean', 'Mild Blur', 'Heavy Blur']
    std_accs    = [std_acc_clean, std_acc_mild, std_acc_heavy]
    mt_accs     = [mt_acc_clean,  mt_acc_mild,  mt_acc_heavy]

    plot_comparison_graphs(blur_levels, std_accs, mt_accs, output_dir=OUTPUT_DIR)
    print_summary_table(blur_levels, std_accs, mt_accs)

    print(f"\n[DONE] All finished! You can view all saved images and models in the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()
