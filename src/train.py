import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time, datetime, os

from src.model.ResNet import ResNet34
from src.model.SimpleNet import SimpleNet, SimpleNetv2
from src.util.DataLoader import getCIFARLoader, getFashionMNISTLoader


def train_network(device, epochs=50, batch_size=128, group_norm=False, dataset="CIFAR10"):
    assert dataset in ["CIFAR10", "FashionMNIST"], "Other datasets are not implemented"
    os.makedirs("model", exist_ok=True)

    norm_str = "GN" if group_norm else "BN"
    model_save_name = f"simplenetv2-epochs{epochs}-bs{batch_size}-{norm_str}-ds{dataset}.pth"

    input_channels = 3 if dataset == "CIFAR10" else 1

    # todo generic model class as param!
    model = SimpleNetv2(num_classes=10, group_norm=group_norm, input_channels=input_channels,
                      input_size=(32, 32) if dataset == "CIFAR10" else (28, 28), groups=16)
    model.to(device)
    model.train()

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_loader = None
    test_loader = None
    if dataset == "CIFAR10":
        train_loader = getCIFARLoader(train=True, batch_size=batch_size)
        test_loader = getCIFARLoader(train=False, batch_size=batch_size)
    else:
        train_loader = getFashionMNISTLoader(train=True, batch_size=batch_size)
        test_loader = getFashionMNISTLoader(train=False, batch_size=batch_size)

    time_now = datetime.datetime.now()
    dt_string = time_now.strftime("%d-%m-%Y-%H-%M")
    modelid_str = f"model_{dt_string}_{model_save_name}"
    writer = SummaryWriter(log_dir=f"runs/{modelid_str}",
                           comment=modelid_str,
                           flush_secs=30)

    best_acc = 0
    print("~~~Starting training loop~~~")

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        for idx, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = data.to(device), label.to(device)

            pred = model(data)

            loss_val = loss(pred, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()
            class_pred = torch.argmax(pred, dim=1)
            correct += (class_pred == label).float().sum()

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        writer.add_scalar("Train/loss", running_loss / idx, epoch)
        writer.add_scalar("Train/acc", 100 * correct / (batch_size * len(train_loader)), epoch)

        running_loss = 0
        correct = 0
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss_val = loss(pred, label)

            running_loss += loss_val.item()
            class_pred = torch.argmax(pred, dim=1)
            correct += (class_pred == label).float().sum()

        curr_acc = 100 * correct / (batch_size * len(test_loader))
        writer.add_scalar("Test/loss", running_loss / idx, epoch)
        writer.add_scalar("Test/acc", curr_acc, epoch)

        print(
            f"Epoch ({epoch}) - Current Loss ({running_loss / idx} - Current Acc. ({curr_acc}) - Best Acc. ({best_acc})")

        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(model.state_dict(), os.path.join("model", f"{model_save_name}_{dt_string}"))

        scheduler.step()


def main():
    print("GPU available:", torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)
    #train_network(device, epochs=30, batch_size=2, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=2, group_norm=True, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=4, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=4, group_norm=True, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=8, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=8, group_norm=True, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=16, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=16, group_norm=True, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=32, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=32, group_norm=True, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=64, group_norm=False, dataset="CIFAR10")
    #train_network(device, epochs=30, batch_size=64, group_norm=True, dataset="CIFAR10")

    train_network(device, epochs=30, batch_size=2, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=2, group_norm=True, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=4, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=4, group_norm=True, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=8, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=8, group_norm=True, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=16, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=16, group_norm=True, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=32, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=32, group_norm=True, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=64, group_norm=False, dataset="FashionMNIST")
    train_network(device, epochs=30, batch_size=64, group_norm=True, dataset="FashionMNIST")


if __name__ == "__main__":
    main()
