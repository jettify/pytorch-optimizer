import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(conf, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % conf.log_interval == 0:
            loss = loss.item()
            idx = batch_idx + epoch * (len(train_loader))
            writer.add_scalar('Loss/train', loss, idx)
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )


def test(conf, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    writer.add_scalar('Accuracy', correct, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)


def prepare_loaders(conf, use_cuda=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


class Config:
    def __init__(
        self,
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 15,
        lr: float = 0.01,
        gamma: float = 0.7,
        no_cuda: bool = True,
        seed: int = 42,
        log_interval: int = 10,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.no_cuda = no_cuda
        self.seed = seed
        self.log_interval = log_interval


def main():
    conf = Config()
    log_dir = 'runs/mnist_custom_optim'
    print('Tensorboard: tensorboard --logdir={}'.format(log_dir))

    with SummaryWriter(log_dir) as writer:
        use_cuda = not conf.no_cuda and torch.cuda.is_available()
        torch.manual_seed(conf.seed)
        device = torch.device('cuda' if use_cuda else 'cpu')
        train_loader, test_loader = prepare_loaders(conf, use_cuda)

        model = Net().to(device)

        # create grid of images and write to tensorboard
        images, labels = next(iter(train_loader))
        img_grid = utils.make_grid(images)
        writer.add_image('mnist_images', img_grid)
        # visualize NN computation graph
        writer.add_graph(model, images)

        # custom optimizer from torch_optimizer package
        optimizer = optim.DiffGrad(model.parameters(), lr=conf.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=conf.gamma)
        for epoch in range(1, conf.epochs + 1):
            train(conf, model, device, train_loader, optimizer, epoch, writer)
            test(conf, model, device, test_loader, epoch, writer)
            scheduler.step()
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram('{}.grad'.format(name), param.grad, epoch)


if __name__ == '__main__':
    main()
