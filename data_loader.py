from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_loader(phase=1):
    val_data = CIFAR10(root='/path/to/your/folder', train=False, download=True, transform=transforms.ToTensor())
    print(val_data.data.shape)
    # print(val_data.targets.shape)
    val_loader = DataLoader(dataset=val_data, pin_memory=True, batch_size=100, drop_last=False,
                            num_workers=1, shuffle=False)
    return val_loader
