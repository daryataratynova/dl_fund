import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#OBSERVATION:lr = 0.2 and epoch = 15 train and val acc ) 98
class MyDataset(Dataset):
    def __init__(self, X, y):

        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index): #get pair x and y
        x = self.features[index]
        y = self.labels[index]        
        return x, y

    def __len__(self):
        return self.labels.shape[0]

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)
    
    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas


def standardize(df, train_mean, train_std):
    return (df - train_mean)/train_std 

def compute_accuracy(model, dataloader, train_mean, train_std):

    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, class_labels) in enumerate(dataloader):

        with torch.no_grad():
            features = standardize(features, train_mean, train_std)
            probas = model(features)
        
        pred = torch.where(probas > 0.5, 1, 0)
        lab = class_labels.view(pred.shape).to(pred.dtype)

        compare = lab == pred
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples

    
df = pd.read_csv("unit3/data_banknote_authentication.txt", header=None)

#Dataset preparation
X_features = df[[0, 1, 2, 3]].values #first 4 columns
y_labels = df[4].values

train_size = int(X_features.shape[0]*0.80) #80 for trining and 20 for validation
val_size = X_features.shape[0] - train_size

dataset = MyDataset(X_features, y_labels)

torch.manual_seed(1)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    dataset=train_set,
    batch_size=10,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val_set,
    batch_size=10,
    shuffle=False,
)

#Standardization
train_mean = torch.zeros(X_features.shape[1])

for x, y in train_loader:
    train_mean += x.sum(dim=0)
    
train_mean /= len(train_set)

train_std = torch.zeros(X_features.shape[1])
for x, y in train_loader:
    train_std += ((x - train_mean)**2).sum(dim=0)

train_std = torch.sqrt(train_std / (len(train_set)-1))



val_mean = torch.zeros(X_features.shape[1])

for x, y in val_loader:
    val_mean += x.sum(dim=0)
    
val_mean /= len(val_set)

val_std = torch.zeros(X_features.shape[1])
for x, y in val_loader:
    val_std += ((x - val_mean)**2).sum(dim=0)

val_std = torch.sqrt(val_std / (len(val_set)-1))

#Training
torch.manual_seed(1)
model = LogisticRegression(num_features=4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2) ## possible SOLUTION

num_epochs = 15 ## possible SOLUTION

for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):

        features = standardize(features, train_mean, train_std) ## SOLUTION
        probas = model(features)
        
        loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 20: # log every 20th batch
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
                   f' | Batch {batch_idx:03d}/{len(train_loader):03d}'
                   f' | Loss: {loss:.2f}')
train_acc = compute_accuracy(model, train_loader, train_mean,train_std)
val_acc = compute_accuracy(model, val_loader, val_mean, val_std)
print(f"Train Accuracy: {train_acc*100:.2f}%, Val Accuracy:{val_acc*100:.2f}% ")