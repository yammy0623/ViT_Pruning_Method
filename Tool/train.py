from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config as cfg

imagenet_data = ImageNet(cfg.imagenet_root)
data_loader = DataLoader(imagenet_data, batch_size=4, shuffle=True)


def get_dataloader(dataset_dir, batch_size=1, split='test'):
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageNet(cfg.imagenet_root, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

# def train_epoch():
#     train_loss_list, val_loss_list = [], []
#     train_acc_list, val_acc_list = [], []
#     for epoch in range(cfg.epochs):
