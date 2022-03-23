import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cov1 = nn.Sequential(                  # (1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16,
            kernel_size=5, stride=1, padding=2,),   # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),            # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.cov2 = nn.Sequential(                  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),             # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                        # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32*7*7, 10)            # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


if __name__ == "__main__":
    with np.load("train_data.npz") as f:
        train_images = torch.unsqueeze(torch.tensor(f['train_images']), dim=1).type(torch.FloatTensor)/255
        train_labels = torch.tensor(np.argmax(f['train_lables'], axis=1))
    with np.load("test_data.npz") as f:
        test_images = torch.tensor(f['test_images'])
        test_labels = torch.tensor(np.argmax(f['test_lables'], axis=1))
    train_data = Data.TensorDataset(train_images, train_labels)
    test_data = Data.TensorDataset(test_images, test_labels)
    test_x = torch.unsqueeze(test_data.tensors[0], dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.tensors[1][:2000]
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    cnn.train()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (pred_y == test_y.data.numpy()).astype(float).sum()/float(test_y.size(0))
                print(f'Epoch: {epoch}| train loss: {loss.data.numpy():.4f} | test accuracy: {accuracy:.2f}')
    torch.save(cnn.state_dict, "cnn.pth")
