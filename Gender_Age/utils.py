# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import all necessary package"""
import torch
from tqdm.auto import tqdm
from typing import Tuple, List
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn_gender: torch.nn.Module,
               loss_fn_age: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[List, List, List[float]]:
    """Trains a PyTorch model for a single epoch.

    Turns target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn_gender: A PyTorch loss function to minimize for gender.
    loss_fn_age: A PyTorch loss function to minimize for age.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss of list gender loss, list age loss and list training accuracy metrics.
    In the form ([train_loss_gender], [train_loss_age], [train_accuracy]). For example:

    ([0.1112,0.5], [0.8743,0.3], [0.23, 0.54])
    """
    # Put model in train mode
    model.train()
    lossgender, lossage, acc_ = [], [], []
    acc_fn = Accuracy('binary').to(device)

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Loop through data loader data batches
    for batch, (X, y, z) in enumerate(loop):
        # Send data to target device
        X = X.to(device)
        y, z = torch.tensor(y, dtype=torch.float, device=device), torch.tensor(z, dtype=torch.float, device=device)

        # 1. Forward pass
        y_pred, z_pred = model(X)
        y_pred = y_pred.squeeze()
        z_pred = z_pred.squeeze()

        # 2. Calculate and accumulate loss
        loss_gender = loss_fn_gender(y_pred, y)
        lossgender.append(loss_gender)
        loss_age = loss_fn_age(z_pred, z)
        lossage.append(loss_age)
        loss = 100 * loss_age + loss_gender

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric
        acc = acc_fn(y_pred, y)
        acc_.append(acc)

    return lossgender, lossage, acc_


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn_gender: torch.nn.Module,
              loss_fn_age: torch.nn.Module,
              device: torch.device) -> Tuple[List, List, List[float]]:
    """Tests(Validation) a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validation dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Setup test loss and test accuracy values
    test_loss_gender, test_loss_age, acc_ = [], [], []
    acc_fn = Accuracy('binary').to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y, z) in enumerate(loop):
            # Send data to target device
            X = X.to(device)
            y, z = torch.tensor(y, dtype=torch.float, device=device), torch.tensor(z, dtype=torch.float, device=device)

            # 1. Forward pass
            y_pred, z_pred = model(X)
            y_pred = y_pred.squeeze()
            z_pred = z_pred.squeeze()

            # 2. Calculate and accumulate loss
            loss_gender = loss_fn_gender(y_pred, y)
            test_loss_gender.append(loss_gender)
            loss_age = loss_fn_age(z_pred, z)
            test_loss_age.append(loss_age)

            # Calculate and accumulate accuracy
            acc = acc_fn(y_pred, y)
            acc_.append(acc)

    # Adjust metrics to get average loss and accuracy per batch
    return test_loss_gender, test_loss_age, acc_

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

def Confusion_Matrix(classes_names, y_pred, targets):

    # (Optional) to show the predict of model and true labels
    confmat = ConfusionMatrix(num_classes=len(classes_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred,
                             target=targets)

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=classes_names,
                                    figsize=(100, 100))

    # Save confusion matrix
    fig.savefig('Confusion_Matrix.png')

