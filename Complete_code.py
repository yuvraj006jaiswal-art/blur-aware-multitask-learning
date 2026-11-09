# ============================================================
#  QUICK FIX — Retrain Standard Model on CLEAN data only
#  Run this after blur_research.py has already run once
#  This gives you a proper comparison for your research paper
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
import os, warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load CIFAR-10 (local copy) ─────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

print("✅ Dataset loaded.")

# ── BlurDataset (same as before) ───────────────────────────
class BlurDataset(Dataset):
    def __init__(self, dataset, blur_level=0):
        self.dataset   = dataset
        self.blur_level = blur_level
        self.blur_transform = {
            0: None,
            1: transforms.GaussianBlur(kernel_size=5,  sigma=1.5),
            2: transforms.GaussianBlur(kernel_size=15, sigma=4.0),
        }
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, class_label = self.dataset[idx]
        if self.blur_level > 0:
            image = self.blur_transform[self.blur_level](image)
        return image, class_label, self.blur_level

NUM_WORKERS = 0 if os.name == 'nt' else 2
BATCH = 64

# ── KEY CHANGE: standard model trains on CLEAN only ────────
loader_train_clean_only = DataLoader(
    BlurDataset(train_data, blur_level=0),   # ← ONLY clean images
    batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS
)

# Test loaders (same as before)
loader_test_clean = DataLoader(BlurDataset(test_data, blur_level=0), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
loader_test_mild  = DataLoader(BlurDataset(test_data, blur_level=1), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
loader_test_heavy = DataLoader(BlurDataset(test_data, blur_level=2), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

# ── Standard model (same architecture) ─────────────────────
class StandardMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base            = mobilenet_v2(weights='DEFAULT')
        self.backbone   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    def forward(self, x):
        x = self.backbone(x); x = self.pool(x); x = x.flatten(1)
        return self.classifier(x)

# ── Multi-Task model (same architecture) ───────────────────
class MultiTaskMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, num_blur_levels=3):
        super().__init__()
        base          = mobilenet_v2(weights='DEFAULT')
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.blur_head  = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_blur_levels))
    def forward(self, x):
        x = self.backbone(x); x = self.pool(x); x = x.flatten(1)
        return self.class_head(x), self.blur_head(x)

# ── Training functions ──────────────────────────────────────
def train_standard(model, loader, epochs=5):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(loader):.4f}")
    print("✅ Standard model training done!")
    return model

def train_multitask(model, loader, epochs=5, blur_weight=0.3):
    model.to(device)
    opt  = optim.Adam(model.parameters(), lr=1e-4)
    ce   = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, class_lbl, blur_lbl in loader:
            imgs      = imgs.to(device)
            class_lbl = class_lbl.to(device)
            blur_lbl  = blur_lbl.to(device)
            opt.zero_grad()
            class_out, blur_out = model(imgs)
            loss = ce(class_out, class_lbl) + blur_weight * ce(blur_out, blur_lbl)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(loader):.4f}")
    print("✅ Multi-task model training done!")
    return model

def evaluate(model, loader, model_type='standard'):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            if model_type == 'standard':
                out = model(imgs)
            else:
                out, _ = model(imgs)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(batch[1].numpy())
    return accuracy_score(labels_all, preds_all) * 100

# ── Train standard model on CLEAN ONLY ─────────────────────
print("\n--- Training Standard MobileNetV2 (clean images only) ---")
standard_model = StandardMobileNetV2()
standard_model = train_standard(standard_model, loader_train_clean_only, epochs=5)

# ── Train multi-task model on MIXED (clean+mild+heavy) ─────
train_mixed = ConcatDataset([
    BlurDataset(train_data, blur_level=0),
    BlurDataset(train_data, blur_level=1),
    BlurDataset(train_data, blur_level=2),
])
loader_train_mixed = DataLoader(train_mixed, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

print("\n--- Training Multi-Task MobileNetV2 (mixed blur, blur-aware) ---")
multitask_model = MultiTaskMobileNetV2()
multitask_model = train_multitask(multitask_model, loader_train_mixed, epochs=5)

# ── Evaluate both ───────────────────────────────────────────
std_accs = [evaluate(standard_model, l, 'standard') for l in [loader_test_clean, loader_test_mild, loader_test_heavy]]
mt_accs  = [evaluate(multitask_model, l, 'multitask') for l in [loader_test_clean, loader_test_mild, loader_test_heavy]]

blur_levels = ['Clean', 'Mild Blur', 'Heavy Blur']

print("\n" + "="*55)
print(f"{'Blur Level':<20} {'Standard':>12} {'Multi-Task':>12}")
print("="*55)
for level, s, m in zip(blur_levels, std_accs, mt_accs):
    diff = m - s
    sign = "+" if diff >= 0 else ""
    print(f"{level:<20} {s:>11.2f}% {m:>11.2f}%   ({sign}{diff:.2f}%)")
print("="*55)

# ── Graphs ──────────────────────────────────────────────────
# Graph 1 — Line graph
plt.figure(figsize=(8, 5))
plt.plot(blur_levels, std_accs, 'o--', color='tomato',    label='Standard (clean-only training)', linewidth=2)
plt.plot(blur_levels, mt_accs,  's-',  color='steelblue', label='Multi-Task (blur-aware training)', linewidth=2)
plt.title('Classification Accuracy vs Blur Level', fontsize=14)
plt.xlabel('Blur Level'); plt.ylabel('Accuracy (%)')
plt.ylim(0, 100); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('line_graph_v2.png', dpi=150); plt.show()
print("✅ Saved line_graph_v2.png")

# Graph 2 — Bar chart
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, std_accs, w, label='Standard (clean-only)',  color='tomato',    alpha=0.85)
b2 = ax.bar(x + w/2, mt_accs,  w, label='Multi-Task (blur-aware)', color='steelblue', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(blur_levels)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Standard vs Multi-Task Model', fontsize=14)
ax.set_ylim(0, 100); ax.legend()
ax.bar_label(b1, fmt='%.1f%%', padding=3)
ax.bar_label(b2, fmt='%.1f%%', padding=3)
plt.tight_layout(); plt.savefig('bar_chart_v2.png', dpi=150); plt.show()
print("✅ Saved bar_chart_v2.png")

# Graph 3 — Accuracy DROP comparison (most powerful for paper)
std_drop  = [std_accs[0] - s for s in std_accs]   # how much standard loses
mt_drop   = [mt_accs[0]  - m for m in mt_accs]    # how much multi-task loses

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(blur_levels, std_drop, 'o--', color='tomato',    label='Standard model accuracy drop', linewidth=2)
ax.plot(blur_levels, mt_drop,  's-',  color='steelblue', label='Multi-Task model accuracy drop', linewidth=2)
ax.set_title('Accuracy Drop from Clean Baseline', fontsize=14)
ax.set_xlabel('Blur Level'); ax.set_ylabel('Accuracy Drop (%)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('accuracy_drop.png', dpi=150); plt.show()
print("✅ Saved accuracy_drop.png  ← MOST POWERFUL GRAPH FOR YOUR PAPER")

torch.save(standard_model.state_dict(),  'standard_model_v2.pth')
torch.save(multitask_model.state_dict(), 'multitask_model_v2.pth')
print("\n✅ All done! Use line_graph_v2.png, bar_chart_v2.png, accuracy_drop.png in your paper.")