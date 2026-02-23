from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_dataset():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5)
                                   ]))
    test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)
                                  ]))
    return train_dataset, test_dataset

def load_mnist_dataloader(batch_size):
    train_dataset, test_dataset = load_mnist_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader