import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import timm
from datetime import datetime
import time
import sys

# from utils import set_seed, write_config_log, write_result_log
import os
import wandb
import yaml

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# CONFIG
# Learning Options
EPOCHS = 50  # train how many epochs
BATCH_SIZE = 128  # batch size for dataloader (256太大了)
USE_ADAM = False  # Adam or SGD optimizer
LR = 1e-3  # learning rate
MILESTONES = [16, 32, 45]  # reduce learning rate at 'milestones' epochs

# Param


class ModelConfig:
    def __init__(
        self,
        exp_name: str,
        model_type: str,
        model_name: str,
        runs: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        self.exp_name = exp_name
        self.model_type = model_type
        self.model_name = model_name
        self.runs = runs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


def write_result_log(
    logfile_path, epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better
):
    """write experiment log file for result of each epoch to ./experiment/{exp_name}/log/result_log.txt"""
    with open(logfile_path, "a") as f:
        f.write(
            f"[{epoch + 1}/{cfg.epochs}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )
        if is_better:
            f.write(" -> val best (acc)")
        f.write("\n")


def plot_learning_curve(logfile_dir, result_lists):
    # 確保資料轉換為 CPU 再繪圖
    train_acc = [
        x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        for x in result_lists["train_acc"]
    ]
    train_loss = [
        x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        for x in result_lists["train_loss"]
    ]
    val_acc = [
        x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        for x in result_lists["val_acc"]
    ]
    val_loss = [
        x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        for x in result_lists["val_loss"]
    ]

    plt.figure()
    plt.plot(train_acc)
    plt.title("Train Accuracy History")
    plt.ylabel("Train Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train accuracy"], loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, "train_acc.jpg"))
    plt.close()

    plt.figure()
    plt.plot(train_loss)
    plt.title("Train Loss History")
    plt.ylabel("Train Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train Loss"], loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, "train_loss.jpg"))
    plt.close()

    plt.figure()
    plt.plot(val_acc)
    plt.title("Validation Accuracy History")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Valid accuracy"], loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, "val_acc.jpg"))
    plt.close()

    plt.figure()
    plt.plot(val_loss)
    plt.title("Validation Loss History")
    plt.ylabel("Valid Loss")
    plt.xlabel("Epoch")
    plt.legend(["Valid Loss"], loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, "val_loss.jpg"))
    plt.close()


def train_model(
    device,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    model_save_dir,
    logfile_dir,
):
    # model.train()  # 設置模型為訓練模式
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    # for early stopping
    best_loss = np.inf
    best_epoch = 0
    counter = 0
    patience = 50
    for epoch in range(epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(
                f"\r[{epoch + 1}/{epochs}] Train batch: {batch + 1} / {len(train_loader)}"
            )
            sys.stdout.flush()
            # Data loading.
            images, labels = data
            images, labels = images.to(device), labels.to(
                device
            )  # (batch_size, 3, 32, 32), (batch_size)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        print()
        print(
            f"[{epoch + 1}/{epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}"
        )
        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            for batch, data in enumerate(val_loader):
                sys.stdout.write(
                    f"\r[{epoch + 1}/{epochs}] Validation batch: {batch + 1} / {len(val_loader)}"
                )
                sys.stdout.flush()

                # Data loading.
                images, labels = data
                images, labels = images.to(device), labels.to(
                    device
                )  # (batch_size, 3, 32, 32), (batch_size)
                # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
                pred = model(images)
                # Calculate loss.
                loss = criterion(pred, labels)
                # Evaluate.
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print()
        print(
            f"[{epoch + 1}/{epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}"
        )

        # Scheduler step
        # scheduler.step()
        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )

        current_result_lists = {
            "train_acc": train_acc_list,
            "train_loss": train_loss_list,
            "val_acc": val_acc_list,
            "val_loss": val_loss_list,
        }

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(
            os.path.join(logfile_dir, "result_log.txt"),
            epoch,
            epoch_time,
            train_acc,
            val_acc,
            train_loss,
            val_loss,
            is_better,
        )
        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f"[{epoch + 1}/{epochs}] Save best model to {model_save_dir} ...")
            torch.save(
                model.state_dict(), os.path.join(model_save_dir, "model_best.pth")
            )
            best_acc = val_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}. Best validation loss: {best_loss:.4f}."
                )
                break

    plot_learning_curve(logfile_dir, current_result_lists)


def test_model(device, model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(
                device
            )  # (batch_size, 3, 32, 32), (batch_size)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"acc: {100 * correct / total}%")


def write_config(cfg: ModelConfig, config_file_path: str):
    config_dict = {
        "Experiment Name": cfg.exp_name,
        "Model Type": cfg.model_type,
        "Model Name": cfg.model_name,
        "Number of Runs": cfg.runs,
        "Epochs": cfg.epochs,
        "Batch Size": cfg.batch_size,
        "Learning Rate": cfg.learning_rate,
    }
    with open(config_file_path, "w") as f:
        yaml.dump(config_dict, f)

    print(f"Configuration saved to {config_file_path}")


def main(cfg):
    print(f"Running experiment {cfg.exp_name} with model {cfg.model_name}")
    wandb.init(
        project=cfg.exp_name,
        config={"learning_rate": cfg.learning_rate, "epoch": cfg.epochs},
        name=cfg.model_name,
    )
    # Experiment name
    exp_name = (
        cfg.model_type
        + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        + "_"
        + cfg.exp_name
    )

    # Write log file for config
    logfile_dir = os.path.join("./experiment", exp_name, "log")
    os.makedirs(logfile_dir, exist_ok=True)
    # Write config
    config_path = os.path.join("./experiment", exp_name, "config.ysaml")
    write_config(cfg, config_path)
    # write_config_log(os.path.join(logfile_dir, 'config_log.txt'))
    model_save_dir = os.path.join("./experiment", exp_name, "model")
    os.makedirs(model_save_dir, exist_ok=True)

    # MODEL
    # (ViT backbone)
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=10)
    # summary(model, input_size=(1, 3, 224, 224))

    # GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_size = model.default_cfg["input_size"]
    print(input_size[1:])

    # DATALOADER
    # Set CIFAR-10 img from to 224x224
    transform_train = transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Load CIFAR-10
    # trainset_full = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform_train
    # )
    # train_size = int(0.7 * len(trainset_full))
    # val_size = len(trainset_full) - train_size
    # trainset, valset = random_split(trainset_full, [train_size, val_size])

    # testset = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform_test
    # )

    # Load ImageNet
    data_dir = "/data/imagenet"
    trainset = datasets.ImageNet(root=f"{data_dir}/train", transform=transform_train)
    valset = datasets.ImageNet(root=f"{data_dir}/val", transform=transform_train)
    testset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform_test)

    # 70% training, 30% val

    # DataLoader
    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    # OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    milestones = [16, 32, 45]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # TRAIN
    train_model(
        device,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        cfg.epochs,
        model_save_dir,
        logfile_dir,
    )
    test_model(device, model, test_loader)
    wandb.finish()


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        configs = yaml.safe_load(file)

    for model_cfg in configs["models"]:
        cfg = ModelConfig(**model_cfg)
        main(cfg)

    # cfg = ModelConfig(
    #     exp_name = "cait",
    #     model_type = "cait",
    #     model_name = "cait_xxs24_224.fb_dist_in1k",
    #     runs = 50,
    #     epochs = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-4
    # )
    # main(cfg)

    # cfg = ModelConfig(
    #     exp_name = "augreg",
    #     model_type = "augreg",
    #     model_name = "vit_small_patch32_224.augreg_in21k_ft_in1k",
    #     runs = 50,
    #     epochs = 200,
    #     batch_size = 64,
    #     learning_rate = 1e-4
    # )
    # main(cfg)

    # cfg = ModelConfig(
    #     exp_name = "deit3",
    #     model_type = "deit3",
    #     model_name = "deit3_small_patch16_384.fb_in22k_ft_in1k",
    #     runs = 50,
    #     epochs = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-4
    # )
    # main(cfg)

    # cfg = ModelConfig(
    #     exp_name = "deit3",
    #     model_type = "deit3",
    #     model_name = "deit3_small_patch16_384.fb_in22k_ft_in1k",
    #     runs = 50,
    #     epochs = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-4
    # )
    # main(cfg)
