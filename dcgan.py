import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self, z_dim):
        super(generator, self).__init__()
        self.net = nn.Sequential(
            # input:(n, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0),  # (n, 256, 4， 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1),  # (n, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (n, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # (n, 1, 28, 28)
            nn.Tanh()
        )
        self.weight_init()

    def forward(self, x):
        x = self.net(x)
        return x

    def weight_init(self):
        for m in self.net.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            # input:(n, 1, 28, 28)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), # (n, 64, 14, 14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # (n, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), # (n, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 2, 0, bias=False), # (n, 1, 1, 1)
            nn.Dropout2d(0.5),
            nn.Sigmoid()
        )
        self.weight_init()

    def forward(self, x):
        x = self.net(x)
        out = x.view(x.size(0), -1)
        return out

    def weight_init(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    netG = generator(z_dim=100)
    random_noise = torch.randn(size=(64, 100, 1, 1))
    fake = netG(random_noise)
    print(fake.size())
