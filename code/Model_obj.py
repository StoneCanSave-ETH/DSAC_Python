import torch.nn as nn
import torch
import numpy as np
import math


# Parameter Setting
storeIntervalPre = 100
lrInterval = 5000
batchSize = 1600


class OBJ_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0, ),
            nn.ReLU(),
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=2*2*512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


def criterion_loss(pred, label):
    return torch.mean(torch.norm(pred - label, 2, 1))


def train(model, trainloader, optimizer, num, device):

    # Set training
    model.train()
    for (BatchData, BatchLabel, pixel, BGR) in trainloader:
        # For cuda use
        BatchData, BatchLabel = BatchData.to(device), BatchLabel.to(device)

        # calculate prediction and loss
        pred = model(BatchData)
        loss = criterion_loss(pred, BatchLabel)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        If we wanna use fire scene, change the following:
        torch.save(model.state_dict(), './Model parameter/obj_model_init.pkl') 
        -> torch.save(model.state_dict(), './Model parameter/fire/obj_model_init.pkl') 
        
        and create a new file in ./Model parameter, called fire, i.e. ./Model parameter/fire
        '''
        # num counter
        num += 1
        if not num % storeIntervalPre:
            torch.save(model.state_dict(), './Model parameter/chess/scene/obj_model_init.pkl')
        if not num % lrInterval:
            for param in optimizer.param_groups:
                param['lr'] *= 0.5
        if not num % 100:
            print('Selected Pixel:', pixel[:2])
            print('Selected Image:', BGR[:2])
            print('One of info:', BatchLabel[0].cpu().detach().numpy())
            print('One of predication:', pred[0].cpu().detach().numpy())
            print('Train Loss:', loss.item())

    return loss.item(), num


def test(model, test_loader, device, num):
    model.eval()
    test_loss = 0
    for idx, (BatchData, BatchLabel, pixel, BGR) in enumerate(test_loader):
        BatchData, BatchLabel = BatchData.to(device), BatchLabel.to(device)
        pred = model(BatchData)
        loss = criterion_loss(pred, BatchLabel)
        test_loss += loss.item()
        num += 1
        if not num % 10:
            print('Selected Pixel:', pixel[:2])
            print('Selected Image:', BGR[:2])
            print('Test Loss:', loss.item())
            print('one of prediction:', pred[2].cpu().detach().numpy())
            print('one of info:', BatchLabel[2].cpu().detach().numpy())
    return loss, num


def forward(model, data, device):
    # From numpy to tensor
    model.to(device)
    model.eval()
    pred = model(data)
    return pred.cpu().detach().numpy()


def forward_tensor(model, data, device):
    model.train()
    # From numpy to tensor
    model.to(device)
    pred = model(data)
    return pred


def weight_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight.data)
        torch.nn.init.xavier_uniform(model.bias.data)




    



