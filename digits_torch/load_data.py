# IMPORTS
import torch
import torchvision

# CUDA
if torch.cuda.is_available():
    device = torch.device("cuda:0")

# TRAINING AND TESTING DATA
# parameters
use_pin = False
mnist_mean = 0.1307
mnist_std = 0.3081
train_batch_size = 64
test_batch_size = 1000

# create train and test dataset objects to hold MNIST data
train_dataset = torchvision.datasets.MNIST(
    root='files/',
    train=True,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mnist_mean, mnist_std)
        ]
    )
)

test_dataset = torchvision.datasets.MNIST(
    root='files/',
    train=False,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mnist_mean, mnist_std)
        ]
    )
)

# create the dataloader objects to feed the network in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    pin_memory=use_pin
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    pin_memory=use_pin
)
