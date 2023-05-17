from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from collections import Counter

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

class MyDataset(Dataset):
    def __init__(self, img, labels, transform=None):

        self.transform = transform
        self.img = torch.tensor(img, dtype=torch.float32) 
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index):
        img = self.img[index]
        img = torch.tensor(img).to(torch.float32)
        img = img/255.
        y = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, y.to(torch.int64)

    def __len__(self):
        return self.labels.shape[0]


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

def compute_accuracy(model, dataloader):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples

if __name__ == '__main__':
    X_train, y_train = load_mnist('unit4/data/', kind='train')
    X_test, y_test = load_mnist('unit4/data/', kind='t10k')

    train_dataset = MyDataset(
            img=X_train,
            labels=y_train,
            transform=None,
        )
    torch.manual_seed(1)
    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,  # want to shuffle the dataset
            num_workers=2,  # number processes/CPUs to use
        )

    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
        )

    test_dataset = MyDataset(
            img=X_test,
            labels=y_test,
            transform=None,
        )
    test_loader = DataLoader(
        dataset=test_dataset,
            batch_size=32,
            shuffle=True,  # want to shuffle the dataset
            num_workers=2,  # number processes/CPUs to use
        )




    torch.manual_seed(1)
    model = PyTorchMLP(num_features=784, num_classes=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 30

    loss_list = []
    train_acc_list, val_acc_list = [], []
    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 250:
                ### LOGGING
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train Loss: {loss:.2f}"
                )
            loss_list.append(loss.item())

        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)
    test_acc = compute_accuracy(model, test_loader)

    print(f"Train Acc {train_acc*100:.2f}%")
    print(f"Val Acc {val_acc*100:.2f}%")
    print(f"Test Acc {test_acc*100:.2f}%")