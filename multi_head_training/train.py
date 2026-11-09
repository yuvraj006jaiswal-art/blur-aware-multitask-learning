import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, class_criterion, blur_criterion, optimizer, device, epoch=0, epochs=1, blur_weight=0.3):
    """
    Runs ONE training epoch for the multi-task model with live progress bar.
    Learns both object classification AND blur level estimation at the same time!
    """
    model.train()  # Activates Dropout
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"  Epoch [{epoch + 1}/{epochs}]", unit="batch")

    for images, class_labels, blur_labels in progress_bar:
        # 1. Send data to GPU or CPU
        images = images.to(device)
        class_labels = class_labels.to(device)
        blur_labels = blur_labels.to(device)

        # 2. Reset gradients
        optimizer.zero_grad()

        # 3. Forward pass: Model outputs both class and blur predictions
        class_out, blur_out = model(images)

        # 4. Calculate individual losses for each task
        loss_class = class_criterion(class_out, class_labels)
        loss_blur = blur_criterion(blur_out, blur_labels)

        # 5. Combine losses: Class task is main, blur task is secondary helper (scaled by blur_weight)
        total_loss = loss_class + (blur_weight * loss_blur)

        # 6. Backward pass & optimize
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # Update progress bar with total loss and breakdown
        avg_so_far = running_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(
            loss=f"{avg_so_far:.4f}",
            c_loss=f"{loss_class.item():.3f}",
            b_loss=f"{loss_blur.item():.3f}"
        )

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train_multitask(model, train_loader, device, epochs=5, lr=1e-4, blur_weight=0.3):
    """
    Full training loop for Multi-Task MobileNetV2.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    blur_criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting Multi-Task MobileNetV2 Training ({epochs} epochs) ---")
    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model, train_loader, class_criterion, blur_criterion, optimizer, device,
            epoch=epoch, epochs=epochs, blur_weight=blur_weight
        )
        print(f"  Completed Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    print("[OK] Multi-task model training complete!")
    return model


def evaluate(model, dataloader, device, desc="Evaluating"):
    """
    Evaluates image classification accuracy (0.0% to 100.0%).
    Shows live progress during evaluation.
    """
    model.eval()  # Disables Dropout
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"  [{desc}]", leave=False, unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Get predictions (we ignore blur_out during classification evaluation)
            class_out, _ = model(images)
            predictions = class_out.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(acc=f"{(correct / total) * 100:.2f}%")

    accuracy = (correct / total) * 100
    return accuracy
