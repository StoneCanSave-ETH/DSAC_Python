import torch.nn as nn
import torch
import time
import numpy as np

storeIntervalPre = 100
lrIntervalPre = 5000


class SCORE_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


def backward_input(model, data, device):
    data, model = data.to(device).requires_grad_(True), model.to(device)
    output = model(data).requires_grad_(True)
    grad = torch.autograd.grad(outputs=output, inputs=data)[0][0][0].cpu().detach().numpy()
    return grad


def backward(model, outputgrad, device, optimizer):
    outputgrad, model = outputgrad.to(device).requires_grad_(True), model.to(device)
    optimizer.zero_grad()
    outputgrad.backward()
    optimizer.step()


def train(model, train_loader, device, lossfunc, optimizer, num):
    model.train()
    time_start = time.time()
    for idx, (BatchData, BatchLabel) in enumerate(train_loader):
        BatchData, BatchLabel = BatchData.to(device, dtype=torch.float), BatchLabel.unsqueeze(1).to(device, dtype=torch.float)

        # Forward
        pred = model(BatchData)
        loss = torch.mean(lossfunc(pred, BatchLabel))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += 1

        # num counter
        if not (idx + 1) % 2:
            print('prediction:', pred[:4].cpu().detach().numpy())
            print('info:', BatchLabel[:4].cpu().detach().numpy())
            print('Loss:', loss.item())
            time_end = time.time()
            print('idx', idx, 'cost:', time_end - time_start)

        '''
                If we wanna use fire scene, change the following:
                torch.save(model.state_dict(), './Model parameter/score_model_init.pkl')
                -> torch.save(model.state_dict(), './Model parameter/fire/score_model_init.pkl')
                and create a new file in ./Model parameter, called fire i.e. ./Model parameter/fire
        '''

        # For updating learningrate and storing model
        if not num % storeIntervalPre:
            torch.save(model.state_dict(), './Model parameter/chess/scene/score_model_init.pkl')
        if not num % lrIntervalPre:
            for param in optimizer.param_groups:
                param['lr'] *= 0.5
    return loss.item(), num


def forward(model, data, device):
    # From numpy to tensor
    data, model = data.to(device).float(), model.to(device)
    pred = model(data)
    return pred.cpu().detach().numpy()


def forward_tensor(model, data, device):
    model.train()
    # From numpy to tensor
    data, model = data.to(device).float(), model.to(device)
    pred = model(data)
    return pred


def weight_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight.data)