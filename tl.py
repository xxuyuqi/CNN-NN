import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms, models

def data_load():
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
    ])
    with np.load("train_data.npz") as f:
        train_images = trans(torch.unsqueeze(torch.tensor(f['train_images'][:10000]), dim=1).type(torch.FloatTensor).repeat(1,3,1,1))
        train_labels = torch.tensor(np.argmax(f['train_lables'], axis=1)[:10000])
    with np.load("test_data.npz") as f:
        test_images = trans(torch.unsqueeze(torch.tensor(f['test_images'][:2000]), dim=1).type(torch.FloatTensor).repeat(1,3,1,1))
        test_labels = torch.tensor(np.argmax(f['test_lables'], axis=1)[:2000])
    train_data = Data.TensorDataset(train_images, train_labels)
    test_data = Data.TensorDataset(test_images, test_labels)
    return train_data, test_data

def train_model(model, dataloaders, test_data, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)
        for step, (x, y) in enumerate(dataloaders):
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = model(test_data.tensors[0])
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (pred_y == test_data.tensors[1].data.numpy()).astype(float).sum()/float(test_data.tensors[1].size(0))
                print(f'Epoch: {epoch}| train loss: {loss.data.numpy():.4f} | test accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    train_data, test_data = data_load()
    train_dataloader = Data.DataLoader(train_data, batch_size=50, shuffle=True)
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    train_model(model, train_dataloader, test_data, loss_func, optimizer, 10)
