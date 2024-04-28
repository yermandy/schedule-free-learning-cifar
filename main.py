"""Train CIFAR10 with PyTorch."""

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms

import models
from adamw_schedulefree import AdamWScheduleFree

try:
    import rich
    from rich import print
    from tqdm.rich import tqdm

except ImportError:
    from tqdm import tqdm


try:
    import wandb

except ImportError:
    wandb = None


@dataclass
class Config:
    lr: float
    optim: str
    epochs: int
    wandb: bool
    cuda: int
    name: str
    scheduler: str
    model: str


def train(epoch, model, loader, criterion, optimizer):
    print("\nEpoch: %d" % epoch)

    device = next(model.parameters()).device
    model.train()
    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.train()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    mean_loss = torchmetrics.MeanMetric().to(device)

    progress = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        mean_loss.update(loss)
        accuracy.update(predicted, targets)

        progress.set_description(f"Loss: {mean_loss.compute():.3f} | Acc: {accuracy.compute() * 100:.3f}%")

    accuracy = accuracy.compute()
    mean_loss = mean_loss.compute()

    # print(f"Train loss: {mean_loss:.3f} and accuracy: {accuracy * 100:.2f}%")

    return {"train/loss": mean_loss, "train/accuracy": accuracy}


@torch.no_grad()
def val(epoch, model, loader, criterion, optimizer):
    device = next(model.parameters()).device

    model.eval()
    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.eval()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    mean_loss = torchmetrics.MeanMetric().to(device)

    progress = tqdm(enumerate(loader), total=len(loader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)

            mean_loss.update(loss)
            accuracy.update(predicted, targets)

            progress.set_description(f"Loss: {mean_loss.compute():.3f} | Acc: {accuracy.compute() * 100:.3f}%")

    accuracy = accuracy.compute()
    mean_loss = mean_loss.compute()

    # print(f"Val loss: {mean_loss:.3f} and accuracy: {accuracy * 100:.2f}%")

    return {"val/loss": mean_loss, "val/accuracy": accuracy}


def build_model(model_name):
    print("==> Building model")
    try:
        model = getattr(models, model_name)()
    except AttributeError:
        raise ValueError(f"Unknown model: {model_name}, available models: {models.__all__}")

    return model


def build_optimizer(model, config: Config):
    print("==> Building optimizer")
    try:
        optim = eval(f"torch.optim.{config.optim}")
        optimizer = optim(model.parameters(), lr=config.lr)
    except AttributeError:
        if config.optim == "AdamWScheduleFree":
            optimizer = AdamWScheduleFree(model.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {config.optim}, see torch.optim for available optimizers")

    return optimizer


def build_scheduler(optimizer, config: Config):
    print("==> Building scheduler")
    if config.scheduler is not None:
        try:
            scheduler = eval(f"torch.optim.lr_scheduler.{config.scheduler}")
            scheduler = scheduler(optimizer, T_max=200)
        except AttributeError:
            raise ValueError(
                f"Unknown scheduler: {config.scheduler}, see torch.optim.lr_scheduler for available schedulers"
            )
    else:
        scheduler = None

    return scheduler


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(**kwargs):
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate")
    parser.add_argument("--optim", default="AdamW", choices=["AdamW", "AdamWScheduleFree"], help="optimizer")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--wandb", "-wb", action="store_true", help="use wandb to log metrics")
    parser.add_argument("--cuda", default=0, type=int, help="cuda device")
    parser.add_argument("--name", default=None, type=str, help="run name")
    parser.add_argument("--scheduler", default=None, type=str, choices=["CosineAnnealingLR"], help="scheduler")
    parser.add_argument("--model", default="ResNet18", type=str, help="model from models ")
    config = parser.parse_args()

    config = Config(**{**vars(config), **kwargs})

    seed_everything(42)

    device = f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu"

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=12, worker_init_fn=seed_worker, generator=g
    )

    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=12)

    criterion = nn.CrossEntropyLoss()

    model = build_model(config.model).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    results = defaultdict(list)

    if wandb:
        wandb.init(
            project="adamw-schedule-free",
            config=vars(config),
            mode="online" if config.wandb else "disabled",
            name=config.name,
        )
        wandb.watch(model)

    for epoch in range(config.epochs):
        train_results = train(epoch, model, train_loader, criterion, optimizer)
        val_results = val(epoch, model, val_loader, criterion, optimizer)

        for key, value in train_results.items():
            results[key].append(value)

        for key, value in val_results.items():
            results[key].append(value)

        lr = optimizer.param_groups[0]["lr"]

        if wandb:
            wandb.log({**train_results, **val_results, "lr": lr})

        if scheduler is not None:
            scheduler.step()

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
