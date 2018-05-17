import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


dropout = torch.nn.Dropout(p=0.60)
relu = torch.nn.LeakyReLU()
pool = nn.MaxPool2d(2, 2)

class ConvRes(nn.Module):
    def __init__(self, insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
            nn.BatchNorm3d(insize),
            nn.Dropout(drate),
            torch.nn.Conv3d(insize, outsize, kernel_size=2, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.math(x)


class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=3, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv3d(insize, outsize, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm3d(outsize),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(pool, pool),
        )
        self.avgpool = torch.nn.AvgPool3d(pool, pool)

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        return x


class SimpleNet3D(nn.Module):
    def __init__(self, num_classes, n_dim):
        super(SimpleNet3D, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cnn1 = ConvCNN(n_dim, 64, kernel_size=3, pool=3, avg=False)
        self.cnn2 = ConvCNN(64, 64, kernel_size=3, pool=2, avg=True)
        self.cnn3 = ConvCNN(64, 512, kernel_size=3, pool=2, avg=True)

        self.res1 = ConvRes(512, 64)

        self.features = nn.Sequential(
            self.cnn1, dropout,
            self.cnn2,
            self.cnn3,
            self.res1,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(8000, (num_classes)),
        )

    #         self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #         print (x.data.shape)
        x = self.classifier(x)
        #         x = self.sig(x)
        return x
