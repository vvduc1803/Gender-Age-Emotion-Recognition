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
from model import EmotionModel

def main():

    # Create tensorboard file to keep track process training
    writer = SummaryWriter(f'runs/')
    step_train = 0
    step_test = 0

    # Setup data and split data
    train_loader, test_loader, class_names, test_targets = Data_Loader(train_dir=config.TRAIN_DIR,
                                                                       test_dir=config.TEST_DIR,
                                                                       train_transform=config.train_transform,
                                                                       test_transform=config.test_transform,
                                                                       batch_size=config.BATCH_SIZE,
                                                                       num_workers=config.NUM_WORKER)

    # Setup all necessary of model
    model = EmotionModel().to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY, betas=(0.5, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    # Load check point
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LR, config.DEVICE)

    # Start training process
    for epoch in range(config.EPOCHS):

        # Training process
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, config.DEVICE)

        # # Testing process
        test_loss, test_acc, y_preds = test_step(model, test_loader, loss_fn, config.DEVICE)

        # Add information to tensorboard
        for loss, acc in zip(train_loss, train_acc):
            writer.add_scalar(f'Training Loss', loss, global_step=step_train)
            writer.add_scalar(f'Training Accuracy', acc, global_step=step_train)
            step_train += 1

        writer.add_scalar(f'Validation Loss', test_loss, global_step=step_test)
        writer.add_scalar(f'Validation Accuracy', test_acc, global_step=step_test)
        step_test += 1

        # Print information
        print(f'Epoch: {epoch + 1}')
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
        print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc * 100:.2f}%')
        print(f'Testing Loss:  {test_loss:.4f}  | Testing Accuracy:  {test_acc * 100:.2f}%')

        # Sve model each 5 epochs
        if (epoch + 1) % 5 == 0 & config.SAVE_MODEL:
            save_checkpoint(model, optimizer, config.CHECKPOINT_FILE)
        print('--------------------------------------------------------------------')

if __name__ == '__main__':
    main()


