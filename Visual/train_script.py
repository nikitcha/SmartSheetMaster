from __future__ import print_function

import os

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from loss import FocalLoss
from notenet import NoteNet
from datagen import ListDataset

import omr_utils

# Prepare Data
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

trainset = ListDataset(root='./data/manual/images', list_file='./data/manual/annotations_train.csv', train=True, transform=transform, input_size=512)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, collate_fn=trainset.collate_fn)

testset = ListDataset(root='./data/manual/images',  list_file='./data/manual/annotations_val.csv', train=False, transform=transform, input_size=512)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, collate_fn=testset.collate_fn)

# Count number of classes
cfile = open('./data/manual/classes.csv', 'r')
classes = cfile.read().splitlines()
nclasses = len(classes)

# Model
net = NoteNet(num_classes=nclasses, num_anchors = 9, multiple = 2)

# Parameters
lr = 0.001
resume = False
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    print("Number of Parameters: {}".format(omr_utils.count_parameters(net)))

criterion = FocalLoss(num_classes=nclasses)
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
optimizer = optim.Adam(net.parameters(), lr=lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs, volatile=True)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        torch.save(state, 'ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
