import torch 
from torchvision import datasets, transforms
from model import SimpleMNISTClassifier, Net
import numpy as np
from tqdm import tqdm
from datetime import datetime

opts = {
    'lr': 1e-3,
    'epochs': 20,
    'batch_size': 64}


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    train_data = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data/", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader
    

def train(save=False):

    train_loader, test_loader = load_dataset()

    model = SimpleMNISTClassifier(input_dim=(1, 28, 28), num_classes=10)
    print(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=3, min_delta=0)

    # The loop
    for epoch in range(20):
        train_loss = []
        train_accuracy = []
        for i, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            # pass data through network
            outputs = model(data)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            train_accuracy.append((predicted == labels).sum().item() / predicted.size(0))

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

        if early_stopper.early_stop(np.mean(test_loss)):
            print("stopping early....")
            break
        print('epoch: {}, train loss: {}, train_accuracy {}, test loss: {}, test accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(test_loss), np.mean(test_accuracy)))
    
    if save:
        torch.save(model.state_dict(), "models/mnist_10_es_{}.pt".format(datetime.now().strftime("%d-%m-%y %H-%M")))


if __name__ == "__main__":
    train(save=True)