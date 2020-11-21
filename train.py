"""
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

import dataset
import utils
from models import FacialModel


def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (1 - (iteration / max_iter)) * (1 - (iteration / max_iter))
    if lr < 1.0e-7:
        lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def accuracy(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        return (y_true == torch.argmax(y_pred, dim=1)).float().mean()
    elif isinstance(y_true, np.ndarray):
        return np.mean(y_true == np.argmax(y_pred, axis=1))
    else:
        raise TypeError(f"Expect `torch.Tensor` or `np.ndarray`, found {type(y_true)}")


def train(args):
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(
            os.path.join(args.log_dir, "train"), flush_secs=1
        )

    transform = utils.Compose(
        [
            utils.Rotation(),
            utils.Crop(),
            utils.Resize((32, 32)),
            utils.IntensityNormalize(),
            utils.ToTensor(),
        ]
    )
    train_loader = dataset.load_data(
        args.data_dir, batch_size=args.batch_size, transform=transform
    )
    print("Train length:", len(train_loader))

    model = FacialModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print_every = 10
    num_train = len(train_loader)
    for epoch in range(args.epochs):
        model.train()

        lr = adjust_learning_rate_poly(optimizer, 1e-3, epoch, args.epochs)

        running_print_loss = 0
        running_print_accuracy = 0
        running_accuracy = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            acc = accuracy(labels, outputs)
            running_print_accuracy += acc.item()
            running_accuracy += acc.item()
            running_print_loss += loss.item()
            running_loss = loss.item()
            if (i + 1) % print_every == 0:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f accuracy: %.3f lr: %.8f"
                    % (
                        epoch + 1,
                        i + 1,
                        running_print_loss / print_every,
                        running_print_accuracy / print_every,
                        lr,
                    )
                )
                running_print_loss = 0
                running_print_accuracy = 0

            # write train loss summaries
            train_logger.add_scalar("loss", running_loss, epoch * num_train + i + 1)

        train_logger.add_scalar("accuracy", running_accuracy / num_train, epoch + 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default="logs", type=str, help="Log dir")
    parser.add_argument("--epochs", default=20, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--data_dir", default="./CK+", help="Path to CK+ dir")

    args = parser.parse_args()
    train(args)
