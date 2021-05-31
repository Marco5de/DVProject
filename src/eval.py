import torch
from tqdm import tqdm

from src.util.DataLoader import getCIFARLoader, getFashionMNISTLoader
from src.model.SimpleNet import SimpleNet, SimpleNetv2


def model_eval(model_path, device, dataset="CIFAR10", batch_size=128, group_norm=False, modelClass=SimpleNetv2):
    assert dataset in ["CIFAR10", "FashionMNIST"], "Other datasets are not implemented"

    input_channels = 3 if dataset == "CIFAR10" else 1

    test_loader = None
    if dataset == "CIFAR10":
        test_loader = getCIFARLoader(train=False, batch_size=batch_size)
    else:
        test_loader = getFashionMNISTLoader(train=False, batch_size=batch_size)

    model = modelClass(num_classes=10, group_norm=group_norm, input_channels=input_channels,
                      input_size=(32, 32) if dataset == "CIFAR10" else (28, 28), groups=16)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()

    correct = 0
    for data, label in tqdm(test_loader, total=len(test_loader)):
        data, label = data.to(device), label.to(device)
        pred = model(data)

        class_pred = torch.argmax(pred, dim=1)
        correct += (class_pred == label).float().sum()

    acc = 100 * correct / (batch_size * len(test_loader))
    return acc.item()


def __main__():
    print("GPU available:", torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    acc = model_eval("model/Backup/CIFAR_Models/epochs30-bs32-BN-dsCIFAR10_2.pth", device, dataset="CIFAR10", group_norm=False)

    print(acc)

if __name__ == "__main__":
    __main__()