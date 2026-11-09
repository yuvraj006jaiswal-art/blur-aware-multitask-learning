# ============================================================
#  FAST VERSION — Runs in 20-30 mins on CPU
#  Key changes vs original:
#    - Images: 96x96 instead of 224x224  (5x faster)
#    - Subset: 10,000 train, 2,000 test  (5x faster)
#    - Epochs: 3 instead of 5            (1.6x faster)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.metrics import accuracy_score
import os, warnings, time
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── SPEED SETTING: change image size here ──────────────────
IMG_SIZE    = 96      # was 224 — this alone makes it ~5x faster
TRAIN_SIZE  = 10000   # out of 50,000 — use subset
TEST_SIZE   = 2000    # out of 10,000
EPOCHS      = 3       # was 5
BATCH       = 64

# ── Load CIFAR-10 ───────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # ← CHANGED from 224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

try:
    train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True,  transform=base_transform)
    test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True,  transform=base_transform)
except:
    train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=False, transform=base_transform)
    test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=base_transform)

# ── Use only a SUBSET to keep training fast ─────────────────
torch.manual_seed(42)   # fixed seed so results are reproducible
train_indices = torch.randperm(len(train_data))[:TRAIN_SIZE]
test_indices  = torch.randperm(len(test_data))[:TEST_SIZE]

train_subset = Subset(train_data, train_indices)
test_subset  = Subset(test_data,  test_indices)

print(f"✅ Dataset loaded. Using {TRAIN_SIZE} train / {TEST_SIZE} test images.")
print(f"   Image size : {IMG_SIZE}x{IMG_SIZE}  (was 224x224)")
print(f"   Epochs     : {EPOCHS}  (was 5)")

# ── BlurDataset ─────────────────────────────────────────────
class BlurDataset(Dataset):
    def __init__(self, dataset, blur_level=0):
        self.dataset    = dataset
        self.blur_level = blur_level
        self.blur_fn = {
            0: None,
            1: transforms.GaussianBlur(kernel_size=5,  sigma=1.5),
            2: transforms.GaussianBlur(kernel_size=15, sigma=4.0),
        }
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.blur_level > 0:
            img = self.blur_fn[self.blur_level](img)
        return img, label, self.blur_level

# num_workers=0 on Windows avoids crashes
NUM_WORKERS = 0 if os.name == 'nt' else 2

# Standard model trains on CLEAN only (this is the key comparison)
loader_std_train = DataLoader(
    BlurDataset(train_subset, blur_level=0),
    batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS
)

# Multi-task model trains on MIXED (clean + mild + heavy)
mixed_train = ConcatDataset([
    BlurDataset(train_subset, blur_level=0),
    BlurDataset(train_subset, blur_level=1),
    BlurDataset(train_subset, blur_level=2),
])
loader_mt_train = DataLoader(mixed_train, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

# Test loaders — 3 separate sets for clean/mild/heavy
loader_test_clean = DataLoader(BlurDataset(test_subset, 0), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
loader_test_mild  = DataLoader(BlurDataset(test_subset, 1), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
loader_test_heavy = DataLoader(BlurDataset(test_subset, 2), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

# ── Visualise blur levels ───────────────────────────────────
def show_blur_samples():
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for level, ax, title in zip([0, 1, 2], axes, ['Clean', 'Mild Blur', 'Heavy Blur']):
        img = BlurDataset(test_subset, blur_level=level)[0][0]
        img = (img * std[:, None, None] + mean[:, None, None]).permute(1,2,0).clamp(0,1).numpy()
        ax.imshow(img); ax.set_title(title); ax.axis('off')
    plt.suptitle("Same image at 3 blur levels", fontsize=12)
    plt.tight_layout()
    plt.savefig('blur_samples.png', dpi=150); plt.show()
    print("✅ Saved blur_samples.png")

show_blur_samples()

# ── Standard MobileNetV2 ────────────────────────────────────
class StandardMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base            = mobilenet_v2(weights='DEFAULT')
        self.backbone   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.backbone(x)).flatten(1))

# ── Multi-Task MobileNetV2 ──────────────────────────────────
class MultiTaskMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, num_blur_levels=3):
        super().__init__()
        base          = mobilenet_v2(weights='DEFAULT')
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.blur_head  = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_blur_levels))
    def forward(self, x):
        x = self.pool(self.backbone(x)).flatten(1)
        return self.class_head(x), self.blur_head(x)

# ── Train functions ─────────────────────────────────────────
def train_standard(model, loader, epochs=EPOCHS):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    ce  = nn.CrossEntropyLoss()
    t0  = time.time()
    for ep in range(epochs):
        model.train(); loss_sum = 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward(); opt.step()
            loss_sum += loss.item()
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch [{ep+1}/{epochs}]  Loss: {loss_sum/len(loader):.4f}  |  {elapsed:.1f} min elapsed")
    print("✅ Standard model done!")
    return model

def train_multitask(model, loader, epochs=EPOCHS, blur_weight=0.3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    ce  = nn.CrossEntropyLoss()
    t0  = time.time()
    for ep in range(epochs):
        model.train(); loss_sum = 0
        for imgs, class_lbl, blur_lbl in loader:
            imgs      = imgs.to(device)
            class_lbl = class_lbl.to(device)
            blur_lbl  = blur_lbl.to(device)
            opt.zero_grad()
            class_out, blur_out = model(imgs)
            loss = ce(class_out, class_lbl) + blur_weight * ce(blur_out, blur_lbl)
            loss.backward(); opt.step()
            loss_sum += loss.item()
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch [{ep+1}/{epochs}]  Loss: {loss_sum/len(loader):.4f}  |  {elapsed:.1f} min elapsed")
    print("✅ Multi-task model done!")
    return model

def evaluate(model, loader, model_type='standard'):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            out  = model(imgs) if model_type == 'standard' else model(imgs)[0]
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(batch[1].numpy())
    return accuracy_score(labels_all, preds_all) * 100

# ── Train Standard (clean only) ─────────────────────────────
print(f"\n--- Training Standard MobileNetV2 (CLEAN only, {TRAIN_SIZE} images) ---")
standard_model = train_standard(StandardMobileNetV2(), loader_std_train)

std_accs = [evaluate(standard_model, l, 'standard')
            for l in [loader_test_clean, loader_test_mild, loader_test_heavy]]

print(f"\n📊 Standard Model: Clean={std_accs[0]:.2f}%  Mild={std_accs[1]:.2f}%  Heavy={std_accs[2]:.2f}%")

# ── Train Multi-Task (mixed blur) ───────────────────────────
print(f"\n--- Training Multi-Task MobileNetV2 (MIXED blur, {TRAIN_SIZE*3} images) ---")
multitask_model = train_multitask(MultiTaskMobileNetV2(), loader_mt_train)

mt_accs = [evaluate(multitask_model, l, 'multitask')
           for l in [loader_test_clean, loader_test_mild, loader_test_heavy]]

print(f"\n📊 Multi-Task Model: Clean={mt_accs[0]:.2f}%  Mild={mt_accs[1]:.2f}%  Heavy={mt_accs[2]:.2f}%")

# ── Print summary table ──────────────────────────────────────
blur_levels = ['Clean', 'Mild Blur', 'Heavy Blur']
print("\n" + "="*58)
print(f"{'Blur Level':<20} {'Standard':>12} {'Multi-Task':>12}  {'Diff':>6}")
print("="*58)
for level, s, m in zip(blur_levels, std_accs, mt_accs):
    diff = m - s
    sign = "+" if diff >= 0 else ""
    print(f"{level:<20} {s:>11.2f}% {m:>11.2f}%  ({sign}{diff:.2f}%)")
print("="*58)

# ── Graph 1: Line graph ─────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(blur_levels, std_accs, 'o--', color='tomato',    linewidth=2, label='Standard (clean-only training)')
plt.plot(blur_levels, mt_accs,  's-',  color='steelblue', linewidth=2, label='Multi-Task (blur-aware training)')
plt.title('Classification Accuracy vs Blur Level', fontsize=14)
plt.xlabel('Blur Level'); plt.ylabel('Accuracy (%)')
plt.ylim(0, 100); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('line_graph.png', dpi=150); plt.show()
print("✅ Saved line_graph.png")

# ── Graph 2: Bar chart ──────────────────────────────────────
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, std_accs, w, label='Standard (clean-only)',   color='tomato',    alpha=0.85)
b2 = ax.bar(x + w/2, mt_accs,  w, label='Multi-Task (blur-aware)', color='steelblue', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(blur_levels)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Standard vs Multi-Task Model Accuracy', fontsize=14)
ax.set_ylim(0, 100); ax.legend()
ax.bar_label(b1, fmt='%.1f%%', padding=3)
ax.bar_label(b2, fmt='%.1f%%', padding=3)
plt.tight_layout(); plt.savefig('bar_chart.png', dpi=150); plt.show()
print("✅ Saved bar_chart.png")

# ── Graph 3: Accuracy DROP (best graph for paper) ───────────
std_drop = [std_accs[0] - s for s in std_accs]
mt_drop  = [mt_accs[0]  - m for m in mt_accs]
plt.figure(figsize=(8, 5))
plt.plot(blur_levels, std_drop, 'o--', color='tomato',    linewidth=2, label='Standard — accuracy lost')
plt.plot(blur_levels, mt_drop,  's-',  color='steelblue', linewidth=2, label='Multi-Task — accuracy lost')
plt.fill_between(blur_levels, std_drop, mt_drop, alpha=0.15, color='green', label='Improvement')
plt.title('Accuracy Drop from Clean Baseline', fontsize=14)
plt.xlabel('Blur Level'); plt.ylabel('Accuracy Drop (%)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('accuracy_drop.png', dpi=150); plt.show()
print("✅ Saved accuracy_drop.png  ← put this in your paper!")

torch.save(standard_model.state_dict(),  'standard_model.pth')
torch.save(multitask_model.state_dict(), 'multitask_model.pth')
print("\n✅ ALL DONE! Your 3 graphs are ready for the research paper.")