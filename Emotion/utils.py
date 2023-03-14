# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import all necessary package"""
import torch

from tqdm.auto import tqdm
from typing import Tuple, List
from torch.utils.data import DataLoader
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[List, List]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Setup train loss and train accuracy values
    train_loss, train_acc = [], []

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(loop):

        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss.append(loss)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc.append((y_pred_class == y).sum().item()/len(y_pred))

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Final tests a PyTorch model in the last epochs.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss, testing accuracy metrics, and all results in test data to build confusion matrix.
    In the form (test_loss, test_accuracy, y_preds). For example:

    (0.0223, 0.8985, [34, 23, 12, 54, 23, ....])
    """
    # Put model in eval mode
    model.eval()

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Setup y predicts list, test loss and test accuracy values
    y_preds = []
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(loop):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass and add results each patch to list y predicts
            y_pred = model(X)
            y_preds.append(y_pred.cpu())

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            y_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((y_pred_labels == y).sum().item()/len(y_pred_labels))

    # Combine predicts of all batchs
    y_preds = torch.cat(y_preds)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, y_preds

def save_checkpoint(model, optimizer, filename="my_checkpoint.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Meaningful when continue train model, avoid load old checkpoint lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

