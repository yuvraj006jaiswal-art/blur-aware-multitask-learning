import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=0, epochs=1):
    """
    Runs ONE single training epoch with a live progress bar.
    """
    model.train()  # Tell PyTorch we are training (activates Dropout)
    running_loss = 0.0

    # Live progress bar for batches in this epoch
    progress_bar = tqdm(dataloader, desc=f"  Epoch [{epoch + 1}/{epochs}]", unit="batch")

    for images, class_labels, _ in progress_bar:
        # 1. Send data to GPU or CPU
        images = images.to(device)
        class_labels = class_labels.to(device)

        # 2. Reset gradients from last step
        optimizer.zero_grad()

        # 3. Forward pass: Model predicts classes
        predictions = model(images)

        # 4. Compute loss: How far were predictions from actual labels?
        loss = criterion(predictions, class_labels)

        # 5. Backward pass: Calculate gradients
        loss.backward()

        # 6. Step optimizer: Update model weights
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar display with running average loss
        avg_so_far = running_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=f"{avg_so_far:.4f}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train_standard(model, train_loader, device, epochs=5, lr=1e-4):
    """
    Trains the standard model for a specified number of epochs.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting Standard MobileNetV2 Training ({epochs} epochs) ---")
    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, epochs=epochs
        )
        print(f"  Completed Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    print("[OK] Standard model training complete!")
    return model


def evaluate(model, dataloader, device, desc="Evaluating"):
    """
    Evaluates the model accuracy on any given dataloader (Clean, Mild, or Heavy).
    Shows live progress during evaluation.
    Returns accuracy percentage (0.0 to 100.0).
    """
    model.eval()  # Tell PyTorch we are evaluating (turns off Dropout)
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"  [{desc}]", leave=False, unit="batch")

    # Turn off gradient calculation to save memory and compute faster
    with torch.no_grad():
        for batch in progress_bar:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)  # Pick the class with highest probability

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Show live accuracy on progress bar
            progress_bar.set_postfix(acc=f"{(correct / total) * 100:.2f}%")

    accuracy = (correct / total) * 100
    return accuracy
