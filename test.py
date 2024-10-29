from tome_test_img import test_model, ModelConfig
import torch, torchvision
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

EPOCHS     = 10           # train how many epochs
BATCH_SIZE = 128          # batch size for dataloader 
USE_ADAM   = False        # Adam or SGD optimizer
LR         = 1e-3         # learning rate
MILESTONES = [16, 32, 45] # reduce learning rate at 'milestones' epochs


cfg = ModelConfig(
    exp_name = "vit_pre",
    model_type = "vit",
    runs = 50,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LR
)

transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
model_name = "vit_base_patch32_224"  # 也可以使用 "vit_base_patch32_224"
model = timm.create_model(model_name, pretrained=True, num_classes=10)  
model.load_state_dict(torch.load('model_best.pth'))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


# GPU setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
test_model(model, test_loader)