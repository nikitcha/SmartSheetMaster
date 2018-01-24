#$$

# Requires Ghostscript
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import csv
import torch.optim as optim
import torch
from torch.autograd import Variable
import pickle

cpath = 'C:/Users/Niki/Source/SmartSheetOMR'
os.chdir(cpath)

import omr_utils as util

#%%
pdffile = os.path.join(cpath, 'pdf/nocturne.pdf')
pngpath = os.path.join(cpath, 'png/')
pngs = util.pdf_to_png(pdffile, 300, pngpath)

sheet = cv2.imread(pngs[1], 0)

#%%
staff_lines = util.get_staff_lines(sheet)
segment = util.get_segment(sheet, staff_lines, segnum=5, hand='left')

#%% Generate images
numimages = 1000
maxclasses = 20
imsize = (512, 512)
numboxes = (8, 8)

ann = util.generate_images(numimages, maxclasses, imsize, numboxes, datadir = './data/manual/')

#%% Get Labels
X_train, Y_train, X_val, Y_val = util.load_data(numboxes = numboxes)

#%%
in_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
train_data = (torch.from_numpy(in_train), torch.from_numpy(Y_train))
in_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
val_data = (torch.from_numpy(in_val), torch.from_numpy(Y_val))

model = util.Net(Y_train.shape[1])
model.double()
print(util.count_parameters(model))
lossfn = torch.nn.MSELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

model = util.train(model, optimizer, train_data, val_data, lossfn, 50)


#%% Load checkpoint and test

model = util.Net(Y_train.shape[1])

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['state_dict'])
model.double()

ix = 20
rec,pred = util.eval_locations(model, train_data[0][[ix],:,:,:], target = None, numboxes=numboxes) # Y_train[[ix],:] #, 

#%% Test Retina Net - from samples

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from notenet import NoteNet
from encoder import DataEncoder

print('Loading model..')
net = NoteNet(num_classes=23, num_anchors = 9, multiple = 2)
checkpoint = torch.load('ckpt.pth', map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['net'])
    
net.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

img = Image.open('./data/manual/images/200.png')
w = h = 320
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))
cfile = open('./data/manual/classes.csv', 'r')
classes = cfile.read().splitlines()

fnt = ImageFont.truetype("arial.ttf", 10)
draw = ImageDraw.Draw(img)
counter = 0
for box in boxes:
    draw.rectangle(list(box), outline='red')
    draw.text((box[0],box[1]-15), classes[labels[counter]], font = fnt)
    counter+=1
img.show()

#%% Test Retina Net - from PDF

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from notenet import NoteNet
from encoder import DataEncoder

print('Loading model..')
net = NoteNet(num_classes=23, num_anchors = 9, multiple = 2)
checkpoint = torch.load('ckpt.pth', map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['net'])
    
net.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

sheet = np.asarray(Image.open('./png/nocturne-002.png').convert('L'))
staff_lines = util.get_staff_lines(sheet)

s = 640
img = Image.fromarray(sheet[240:240+s,280:280+s])
    
w = h = 320
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))
cfile = open('./data/classes.csv', 'r')
classes = cfile.read().splitlines()

fnt = ImageFont.truetype("arial.ttf", 10)
draw = ImageDraw.Draw(img)
counter = 0
for box in boxes:
    draw.rectangle(list(box), outline='red')
    draw.text((box[0],box[1]-15), classes[labels[counter]], font = fnt)
    counter+=1
img.show()


