import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
import tome
from tqdm import tqdm
import time
import yaml


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


def benchmark_with_dataset(
    model: torch.nn.Module,
    dataset_loader: DataLoader,
    device: torch.device = 0,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:

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
            for i, (inputs, labels) in enumerate(
                tqdm(dataset_loader, disable=not verbose, desc="Benchmarking")
            ):

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


import os


def parse_experiment_folders(experiment_dir):
    if not os.path.exists(experiment_dir):
        print(f"Error: {experiment_dir} does not exist!")
        return []
    first_level_folders = [f.name for f in os.scandir(experiment_dir) if f.is_dir()]

    return first_level_folders


def main(file_folder):
    key_mapping = {
        "Experiment Name": "exp_name",
        "Model Type": "model_type",
        "Model Name": "model_name",
        "Number of Runs": "runs",
        "Epochs": "epochs",
        "Batch Size": "batch_size",
        "Learning Rate": "learning_rate",
    }

    with open(file_folder + "/config.yaml", "r") as file:
        raw_config = yaml.safe_load(file)
    mapped_config = {key_mapping[k]: v for k, v in raw_config.items()}
    cfg = ModelConfig(**mapped_config)
    print(cfg)

    best_model_path = file_folder + "/model/model_best.pth"
    # Load pretrain model
    model_name = cfg.model_name
    model = timm.create_model(model_name, pretrained=False, num_classes=10)

    # Based on pretrained model
    input_size = model.default_cfg["input_size"]
    print(input_size[1:])

    # Dataloader
    transform_test = transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    # Put model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    throughput, accuracy = benchmark_with_dataset(
        model, test_loader, device=device, runs=50, verbose=True
    )
    print(
        f"Benchmark Results -> Throughput: {throughput:.2f} im/s, Accuracy: {accuracy:.2f}%"
    )

    output_file = file_folder + "/benchmark.txt"
    with open(output_file, "w") as file:
        file.write(f"Original\n")
        file.write(f"Throughput: {throughput:.2f} im/s\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n\n")

        if cfg.exp_name != "CNN_based":
            print("Apply ToMe")
            tome.patch.timm(model)
            throughput_tome, accuracy_tome = benchmark_with_dataset(
                model, test_loader, device=device, runs=50, verbose=True
            )
            print(
                f"Benchmark Results -> Throughput: {throughput_tome:.2f} im/s, Accuracy: {accuracy_tome:.2f}%"
            )
            file.write(f"Apply ToMe\n")
            file.write(f"Throughput: {throughput_tome:.2f} im/s\n")
            file.write(f"Accuracy: {accuracy_tome:.2f}%\n\n")


if __name__ == "__main__":
    # experiment_dir = './experiment'
    # folders = parse_experiment_folders(experiment_dir)

    # for folder in folders:
    #     print(folder)
    #     file_folder = os.path.join(experiment_dir, folder)
    #     main(file_folder)

    file_folder = "./experiment/EfficientNet_2024_11_19_21_59_13_CNN_based"
    main(file_folder)
