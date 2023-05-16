import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


#OBSERVATION: with lr = 0.05 and epochs = 10, train acc = 98.72 val acc= 98.91
# with lower lr train acc decreases (# of epochs stays the same)
# with lr = 0.5 and apochs = 10 trains acc = 95.54 val acc = 99.27

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
    
def compute_accuracy(model, dataloader):

    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, class_labels) in enumerate(dataloader):
        
        with torch.no_grad():
            probas = model(features)
        
        pred = torch.where(probas > 0.5, 1, 0)
        lab = class_labels.view(pred.shape).to(pred.dtype)

        compare = lab == pred
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples
    
df = pd.read_csv("data_banknote_authentication.txt", header=None)

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


#Training
torch.manual_seed(1)
model = LogisticRegression(num_features=4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) ## FILL IN VALUE

num_epochs = 10  ## FILL IN VALUE

for epoch in range(num_epochs):
    
    model = model.train() #good practice
    for batch_idx, (features, class_labels) in enumerate(train_loader):

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
train_acc = compute_accuracy(model, train_loader)
val_acc = compute_accuracy(model, val_loader)
print(f"Train Accuracy: {train_acc*100:.2f}%, Val Accuracy:{val_acc*100:.2f}% ")