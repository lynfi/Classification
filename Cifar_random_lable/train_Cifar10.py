"""Training NN on MNIST"""

from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description="Pytorch MNIST Training")

# optimization setting
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optim', default='SGD')

# model and training setting
parser.add_argument('--model', default='VGG19')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--bs', type=int, default=128)

# resume
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from the chekpoint')

# augmentation (random rotation and scale)
parser.add_argument('--aug', action='store_true', help='rotaion and scale')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_worker = 8 if torch.cuda.is_available() else 0

best_acc = 0  # best accuracy
star_epoch = 0  #start from the 0 or last checkpoint

# Data
print('==> Preparing data..')
## The order of transforms matter
## Data augmentation will be done at each batch
if args.aug:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10, (0.05, 0.05), (0.9, 1.1), 10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.bs,
                                          shuffle=True,
                                          num_workers=num_worker)

testset = torchvision.datasets.CIFAR10(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=num_worker)

# Model
print('==> Building model..')
net = VGG(args.model)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # parallel
    torch.backends.cudnn.benchmark = True  # speed up

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no dir found!'
    checkpoint = torch.load('./checkpoint/' + args.net + 'ckp.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    star_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=0)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d|%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving the best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.model + 'ckp.pth')
        best_acc = acc


list_loss = []

for epoch in range(star_epoch, star_epoch + args.epochs):
    trainloss = train(epoch)
    test(epoch)
    list_loss.append(trainloss)
    print(list_loss)
