# ============================================================
#  Grad-CAM Visualization for Standard vs Multi-Task Models
#  Run this in the same folder as standard_model.pth and
#  multitask_model.pth
# ============================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Subset
import cv2
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMG_SIZE = 96
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ── Same model definitions as training ──────────────────────
class StandardMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base            = mobilenet_v2(weights='DEFAULT')
        self.backbone   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.backbone(x)).flatten(1))

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

# ── Load saved models ────────────────────────────────────────
print("Loading models...")
std_model = StandardMobileNetV2()
std_model.load_state_dict(torch.load('standard_model.pth', map_location=device))
std_model.to(device).eval()

mt_model = MultiTaskMobileNetV2()
mt_model.load_state_dict(torch.load('multitask_model.pth', map_location=device))
mt_model.to(device).eval()
print("✅ Models loaded.")

# ── Grad-CAM implementation ──────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer, model_type='standard'):
        self.model       = model
        self.model_type  = model_type
        self.gradients   = None
        self.activations = None

        # Register hooks on the target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        """
        x          : (1, C, H, W) tensor on device
        class_idx  : which class to explain; None → argmax (predicted class)
        returns    : heatmap as numpy array (H, W), range [0,1]
        """
        self.model.zero_grad()
        output = self.model(x)

        # Handle multi-task output tuple
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop w.r.t. the chosen class score
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dims → (C,)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def cam_to_heatmap(cam, img_size):
    """Resize CAM and convert to RGB heatmap."""
    cam_resized = cv2.resize(cam, (img_size, img_size))
    heatmap = cm.jet(cam_resized)[:, :, :3]   # drop alpha → (H, W, 3)
    return heatmap

def overlay(original_img, heatmap, alpha=0.45):
    """Blend original image with heatmap."""
    return np.clip(alpha * heatmap + (1 - alpha) * original_img, 0, 1)


# ── Load test data ───────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
mean = torch.tensor([0.485, 0.456, 0.406])
std  = torch.tensor([0.229, 0.224, 0.225])

try:
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True,  transform=base_transform)
except:
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=base_transform)

blur_mild  = transforms.GaussianBlur(kernel_size=5,  sigma=1.5)
blur_heavy = transforms.GaussianBlur(kernel_size=15, sigma=4.0)


def get_sample(idx=0):
    """Return clean tensor + denormalised numpy image for one test sample."""
    img_tensor, label = test_data[idx]
    img_np = (img_tensor * std[:, None, None] + mean[:, None, None]).permute(1,2,0).clamp(0,1).numpy()
    return img_tensor, img_np, label


# ── Target layer: last conv block of MobileNetV2 ─────────────
# backbone[18] is the final convolutional layer in MobileNetV2
std_gradcam = GradCAM(std_model, std_model.backbone[18], 'standard')
mt_gradcam  = GradCAM(mt_model,  mt_model.backbone[18],  'multitask')


# ── Main plot: 4 samples × 3 blur levels × 2 models ─────────
#    Rows = clean / mild / heavy
#    Columns = pairs of (std overlay, mt overlay) for each sample
NUM_SAMPLES = 4   # pick 4 different test images
SAMPLE_IDS  = [0, 7, 15, 42]   # change these to any test-set index you like

fig, axes = plt.subplots(
    nrows=3,
    ncols=NUM_SAMPLES * 2,
    figsize=(NUM_SAMPLES * 5, 11)
)
fig.suptitle("Grad-CAM: Standard vs Multi-Task Model\n(each pair: Standard | Multi-Task)",
             fontsize=14, fontweight='bold', y=1.01)

blur_labels = ['Clean', 'Mild Blur', 'Heavy Blur']
blur_fns    = [None, blur_mild, blur_heavy]

for col_pair, sample_idx in enumerate(SAMPLE_IDS):
    img_tensor, img_np, true_label = get_sample(sample_idx)

    for row, (blur_label, blur_fn) in enumerate(zip(blur_labels, blur_fns)):
        # Apply blur to tensor if needed
        if blur_fn is not None:
            blurred = blur_fn(img_tensor)
        else:
            blurred = img_tensor.clone()

        blurred_np = (blurred * std[:, None, None] + mean[:, None, None]).permute(1,2,0).clamp(0,1).numpy()
        x = blurred.unsqueeze(0).to(device)

        # --- Standard model ---
        cam_std, pred_std = std_gradcam.generate(x)
        hm_std  = cam_to_heatmap(cam_std, IMG_SIZE)
        ov_std  = overlay(blurred_np, hm_std)

        # --- Multi-task model ---
        cam_mt, pred_mt = mt_gradcam.generate(x)
        hm_mt  = cam_to_heatmap(cam_mt, IMG_SIZE)
        ov_mt  = overlay(blurred_np, hm_mt)

        ax_std = axes[row, col_pair * 2]
        ax_mt  = axes[row, col_pair * 2 + 1]

        ax_std.imshow(ov_std)
        ax_mt.imshow(ov_mt)

        ax_std.axis('off')
        ax_mt.axis('off')

        # Titles only on top row
        if row == 0:
            ax_std.set_title(f"Sample {sample_idx}\nSTD  pred:{CIFAR10_CLASSES[pred_std]}", fontsize=8)
            ax_mt.set_title(f"Sample {sample_idx}\n MT  pred:{CIFAR10_CLASSES[pred_mt]}",  fontsize=8)

        # Row labels on leftmost column
        if col_pair == 0:
            ax_std.set_ylabel(blur_label, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved gradcam_comparison.png")


# ── Bonus: single sample deep-dive (clean image only) ────────
#    Shows: original | CAM heatmap | overlay  for each model
sample_idx = SAMPLE_IDS[0]
img_tensor, img_np, true_label = get_sample(sample_idx)
x = img_tensor.unsqueeze(0).to(device)

cam_std, pred_std = std_gradcam.generate(x)
cam_mt,  pred_mt  = mt_gradcam.generate(x)

hm_std = cam_to_heatmap(cam_std, IMG_SIZE)
hm_mt  = cam_to_heatmap(cam_mt,  IMG_SIZE)

fig2, axes2 = plt.subplots(2, 3, figsize=(10, 7))
fig2.suptitle(f"Grad-CAM Deep-Dive  |  True: {CIFAR10_CLASSES[true_label]}", fontsize=13)

rows_data = [
    ("Standard",   img_np, hm_std, overlay(img_np, hm_std), pred_std),
    ("Multi-Task", img_np, hm_mt,  overlay(img_np, hm_mt),  pred_mt),
]
col_titles = ['Original', 'CAM Heatmap', 'Overlay']

for r, (name, orig, hm, ov, pred) in enumerate(rows_data):
    for c, (img_show, title) in enumerate(zip([orig, hm, ov], col_titles)):
        axes2[r, c].imshow(img_show)
        axes2[r, c].axis('off')
        if r == 0:
            axes2[r, c].set_title(title, fontsize=11)
    axes2[r, 0].set_ylabel(f"{name}\npred: {CIFAR10_CLASSES[pred]}", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('gradcam_deep_dive.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved gradcam_deep_dive.png  ← best single figure for the paper!")
print("\n✅ ALL DONE!")