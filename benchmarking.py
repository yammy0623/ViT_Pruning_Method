import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
import tome
from tqdm import tqdm
import time


class ModelConfig:
    def __init__(self, exp_name: str, model_type: str, best_model_path: str, runs: int, epochs: int, batch_size: int, learning_rate: float):
        self.exp_name = exp_name
        self.model_type = model_type
        self.best_model_path = best_model_path,
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
        epochs = 50,
        batch_size = 128,
        learning_rate = 1e-3
    )

    isToMe = False
    # Best Model Path
    best_model_path = 'model_best.pth'
    # Load pretrain model 
    model_name = "vit_base_patch32_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=10)  

    # Based on pretrained model
    input_size = model.default_cfg["input_size"]
    print(input_size[1:])

    # Dataloader
    transform_test = transforms.Compose([
        transforms.Resize(input_size[1:]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Put model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(cfg.best_model_path, map_location=device, weights_only=True))
    
    
    if isToMe:
        print("\nApply ToMe\n")
        tome.patch.timm(model)
        
    print("\nBenchmarking\n")
    throughput, accuracy = benchmark_with_dataset(model, test_loader, device=device, runs=50, verbose=True)
    print(f"Benchmark Results -> Throughput: {throughput:.2f} im/s, Accuracy: {accuracy:.2f}%")
