import torchvision.transforms as transforms
import torchvision
import torch.utils.data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

CLASSES_CIFAR10 = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def getCIFARLoader(train, batch_size):
    augmentations = None
    if train:
        augmentations = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=augmentations)

    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)


CLASSES_FASHIONMNIST = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                        "Ankle Boot"]


def getFashionMNISTLoader(train, batch_size):
    augmentations = None
    if train:
        augmentations = transforms.Compose([
            # transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        augmentations = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=augmentations)

    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)