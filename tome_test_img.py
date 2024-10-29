import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import timm
import tome
from tqdm import tqdm
from datetime import datetime
import time
import sys
# from utils import set_seed, write_config_log, write_result_log
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# CONFIG
# Learning Options
EPOCHS     = 10           # train how many epochs
BATCH_SIZE = 128          # batch size for dataloader 
USE_ADAM   = False        # Adam or SGD optimizer
LR         = 1e-3         # learning rate
MILESTONES = [16, 32, 45] # reduce learning rate at 'milestones' epochs

# Param

class ModelConfig:
    def __init__(self, exp_name: str, model_type: str, runs: int, epochs: int, batch_size: int, learning_rate: float):
        self.exp_name = exp_name
        self.model_type = model_type
        self.runs = runs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

def write_result_log(logfile_path, epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better):
    ''' write experiment log file for result of each epoch to ./experiment/{exp_name}/log/result_log.txt '''
    with open(logfile_path, 'a') as f:
        f.write(f'[{epoch + 1}/{cfg.epochs}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
        if is_better:
            f.write(' -> val best (acc)')
        f.write('\n')

def plot_learning_curve(logfile_dir, result_lists):
    plt.figure()
    plt.plot(result_lists['train_acc'])
    plt.title('Train Accuracy History')
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epoch')
    # plt.legend(['Train accuracy'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, 'train_acc.jpg'))
    plt.close()

    plt.figure()
    plt.plot(result_lists['train_loss'])
    plt.title('Train Accuracy History')
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train Loss'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, 'train_loss.jpg'))
    plt.close()

    plt.figure()
    plt.plot(result_lists['val_acc'])
    plt.title('Validation Accuracy History')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    # plt.legend(['Valid accuracy'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, 'val_acc.jpg'))
    plt.close()


    plt.figure()
    plt.plot(result_lists['val_loss'])
    plt.title('Validation Loss History')
    plt.ylabel('Valid Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Valid Loss'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(logfile_dir, 'val_loss.jpg'))
    plt.close()

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, model_save_dir, logfile_dir):
    # model.train()  # 設置模型為訓練模式
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    for epoch in range(epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{epochs}] Train batch: {batch + 1} / {len(train_loader)}')
            sys.stdout.flush()
            # Data loading.
            images, labels = data
            images, labels = images.to(device), labels.to(device) # (batch_size, 3, 32, 32), (batch_size)
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
        print(f'[{epoch + 1}/{epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')
        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            for batch, data in enumerate(val_loader):
                sys.stdout.write(f'\r[{epoch + 1}/{epochs}] Validation batch: {batch + 1} / {len(val_loader)}')
                sys.stdout.flush()

                # Data loading.
                images, labels = data
                images, labels = images.to(device), labels.to(device) # (batch_size, 3, 32, 32), (batch_size)
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
        print(f'[{epoch + 1}/{epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'), epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

def test_model(model, testloader):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # (batch_size, 3, 32, 32), (batch_size)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'acc: {100 * correct / total}%')


# 基準測試 (Throughput)
def benchmark_with_dataset(
    model: torch.nn.Module,
    dataset_loader: DataLoader,
    device: torch.device = 0,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with a real dataset loader (e.g. CIFAR-10), calculating throughput and accuracy.

    Args:
     - model: the module to benchmark
     - dataset_loader: the DataLoader for the dataset (e.g. CIFAR-10)
     - device: the device to use for benchmarking
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - throughput measured in images / second
     - accuracy of the model on the dataset
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    correct = 0
    total = 0
    warm_up = int(runs * throw_out)
    total_images = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(dataset_loader, disable=not verbose, desc="Benchmarking")):
                
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total_images = 0
                    correct = 0
                    total = 0
                    start = time.time()

                inputs, labels = inputs.to(device), labels.to(device)
                if use_fp16:
                    inputs = inputs.half()

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                total_images += inputs.size(0)

                if i >= runs:  # Limit the number of runs
                    break

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total_images / elapsed
    accuracy = 100 * correct / total

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
        print(f"Accuracy: {accuracy:.2f}%")

    return throughput, accuracy

if __name__ == '__main__':
    cfg = ModelConfig(
        exp_name = "vit_pre",
        model_type = "vit",
        runs = 50,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        learning_rate = LR
    )
    # Experiment name
    exp_name = cfg.model_type + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    # write_config_log(os.path.join(logfile_dir, 'config_log.txt'))
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    # MODEL
    # (ViT backbone)
    model_name = "vit_base_patch32_224"  # 也可以使用 "vit_base_patch32_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=10)  

    # GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # DATALOADER
    # Set CIFAR-10 img from to 224x224
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    # Load CIFAR-10
    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # 80% training, 20% val
    train_size = int(0.6 * len(trainset_full))
    val_size = len(trainset_full) - train_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


    # OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    milestones = [16, 32, 45]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # TRAIN
    train_model(model, train_loader, val_loader, optimizer, criterion, cfg.epochs, model_save_dir, logfile_dir)
    test_model(model, test_loader)

    # print("benchmarking")
    # throughput, accuracy = benchmark_with_dataset(model, dataset_loader, device=device, runs=50, verbose=True)

    # print(f"Benchmark Results -> Throughput: {throughput:.2f} im/s, Accuracy: {accuracy:.2f}%")
    
    
    
