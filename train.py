import torch 
from torchvision import datasets, transforms
from model import SimpleMNISTClassifier
import numpy as np
from tqdm import tqdm
from datetime import datetime

opts = {
    'lr': 1e-3,
    'epochs': 20,
    'batch_size': 64}

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    train_data = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data/", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    return train_loader, test_loader
    

def train(save=False):

    train_loader, test_loader = load_dataset()

    model = SimpleMNISTClassifier(input_dim=(1, 28, 28), num_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # The loop
    for epoch in range(10):
        train_loss = []
        for i, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            # pass data through network
            outputs = model(data)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        test_loss = []
        test_accuracy = []
        for i, (data, labels) in enumerate(test_loader):
            model.eval()
            # pass data through network
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
        print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_accuracy)))
    
    if save:
        torch.save(model.state_dict(), "models/mnist_10_{}.pt".format(datetime.now().strftime("%d-%m-%y %H-%M")))


if __name__ == "__main__":
    train(save=True)