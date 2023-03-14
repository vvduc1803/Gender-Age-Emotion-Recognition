# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import all necessary package"""
import config
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from dataset import Data_Loader
from utils import train_step, test_step, load_checkpoint, save_checkpoint
from model import AgeModel

def main():

    # Create tensorboard file to keep track process training
    writer = SummaryWriter(f'runss/')
    step_train = 0
    step_val = 0

    # Setup data and split data
    train_loader, val_loader = Data_Loader(data_root=config.DATASET,
                                           train_size=0.9,
                                           batch_size=config.BATCH_SIZE,
                                           num_workers=config.NUM_WORKER)

    # Setup all necessary of model
    model = AgeModel().to(config.DEVICE)

    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    loss_fn_gender = nn.BCEWithLogitsLoss()
    loss_fn_age = nn.L1Loss()

    # Load check point
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LR, config.DEVICE)

    # Start training process
    for epoch in range(config.EPOCHS):

        # Training process
        train_loss_gender, train_loss_age, train_acc = train_step(model, train_loader, loss_fn_gender, loss_fn_age,
                                                                  optimizer, config.DEVICE)

        # Validation process
        val_loss_gender, val_loss_age, val_acc = test_step(model, val_loader, loss_fn_gender, loss_fn_age,
                                                           config.DEVICE)

        # Add information to tensorboard
        for loss_gender, loss_age, acc in zip(train_loss_gender, train_loss_age, train_acc):
            writer.add_scalar(f'Training Gender Loss', loss_gender, global_step=step_train)
            writer.add_scalar(f'Training Age Loss', loss_age, global_step=step_train)
            writer.add_scalar(f'Training Accuracy', acc, global_step=step_train)
            step_train += 1

        val_loss_gender = sum(val_loss_gender) / len(val_loss_gender)
        val_loss_age = sum(val_loss_age) / len(val_loss_age)
        val_acc = sum(val_acc) / len(val_acc)

        writer.add_scalar(f'Validation Gender Loss', val_loss_gender, global_step=step_val)
        writer.add_scalar(f'Validation Age Loss', val_loss_age, global_step=step_val)
        writer.add_scalar(f'Validation Accuracy', val_acc, global_step=step_val)
        step_val += 1

        # Print information
        print(f'Epoch: {epoch + 1}')
        train_loss_gender = sum(train_loss_gender) / len(train_loss_gender)
        train_loss_age = sum(train_loss_age) / len(train_loss_age)
        train_acc = sum(train_acc) / len(train_acc)
        print(f'Training Gender Loss: {train_loss_gender:.4f} | Training Age Loss: {train_loss_age:.4f} |'
              f' Training Accuracy: {train_acc * 100:.2f}%')

        print(f'Validation Gender Loss: {val_loss_gender:.4f} | Validation Age Loss: {val_loss_age:.4f} |'
              f' Validation Accuracy: {val_acc * 100:.2f}%')

        # Sve model each 5 epochs
        a = int(input())
        if a == 1 & config.SAVE_MODEL:
            save_checkpoint(model, optimizer, config.CHECKPOINT_FILE)
        print('--------------------------------------------------------------------')

if __name__ == '__main__':
    main()


