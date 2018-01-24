import subprocess
import os
import traceback
import sys
import glob
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import csv
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

# Absolute path to Ghostscript executable or command name if Ghostscript is in PATH.
GHOSTSCRIPTCMD = "gswin64"

class structtype():
    pass

def pdf_to_png(pdffilepath, resolution, pngpath=''):
    if not os.path.isfile(pdffilepath):
        print("'%s' is not a file. Skip." % pdffilepath)

    pdffiledir = os.path.dirname(pdffilepath)
    if not pngpath:
        pngpath = pdffiledir
    
    pdffilename = os.path.basename(pdffilepath)
    pdfname, ext = os.path.splitext(pdffilename)

    try:    
        # Change the "-rXXX" option to set the PNG's resolution.
        # http://ghostscript.com/doc/current/Devices.htm#File_formats
        # For other commandline options see
        # http://ghostscript.com/doc/current/Use.htm#Options
        arglist = [GHOSTSCRIPTCMD,
                  "-q",                     
                  "-dQUIET",                   
                  "-dPARANOIDSAFER",                    
                  "-dBATCH",
                  "-dNOPAUSE",
                  "-dNOPROMPT",                  
                  "-sOutputFile=" + os.path.join(pngpath, pdfname) + "-%03d.png",
                  "-sDEVICE=png16m",                  
                  "-r%s" % resolution,
                  pdffilepath]
        print("Running command:\n%s" % ' '.join(arglist))
        sp = subprocess.Popen(
            args=arglist,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except OSError:
        sys.exit("Error executing Ghostscript ('%s'). Is it in your PATH?" %
            GHOSTSCRIPTCMD)            
    except:
        print("Error while running Ghostscript subprocess. Traceback:")
        print("Traceback:\n%s"%traceback.format_exc())

    stdout, stderr = sp.communicate()
    print("Ghostscript stdout:\n'%s'" % stdout)
    if stderr:
        print("Ghostscript stderr:\n'%s'" % stderr)
    
    return glob.glob(os.path.join(pngpath, pdfname) + '*.png')

def get_staff_lines(sheet):
    # Perform Gaussian Blur
    #kernel = np.ones((3,3),np.float32)/9
    #img = cv2.filter2D(sheet,-1,kernel).astype('float32')/255
    img = sheet.astype('float32')/255
    
    # Find Staff Lines
    scan_line = 1-np.mean(img, axis=1)
    scan_peak_thresh = np.mean(scan_line) + 2.5*np.std(scan_line)
    scan_filtered = scan_line>scan_peak_thresh
    
    scan_peak_locs = np.where(scan_filtered)[0]
    whitespace_widths = np.append(scan_peak_locs,1)-np.append(0, scan_peak_locs)
    whitespace_widths = whitespace_widths[:-1]
    
    
    staff = []
    for i in range(len(whitespace_widths)):
        if whitespace_widths[i]>4:
            staff.append({'position': scan_peak_locs[i], 'start': scan_peak_locs[i], 'end': scan_peak_locs[i], 'segment': 1})
        else:
            staff[-1]['end'] = scan_peak_locs[i]
            staff[-1]['position'] = (staff[-1]['start']+staff[-1]['end'])/2
    
    inter_staff = np.where(scan_line<np.percentile(scan_line,10))[0]
    counter=1
    for i in range(len(staff)):
        if any((staff[i-1]['position']<inter_staff) & (staff[i]['position']>inter_staff)):
            counter+=1
        staff[i]['segment'] = counter

    return staff

def get_segment(sheet, staff_lines, segnum=1, hand='left'):
    
    segments = np.asarray([s['segment'] for s in staff_lines])
    idx = np.where(segments==segnum)[0]
    n = len(idx)
    offset = 60
    
    if n==5:
        if idx[0]==0:
            six = staff_lines[idx[0]]['position'] - offset
        else:
            six = (staff_lines[idx[0]]['position']+staff_lines[idx[0]-1]['position'])/2
        if idx[4]==len(staff_lines):
            eix = staff_lines[idx[4]]['position'] + offset
        else:
            eix = (staff_lines[idx[4]]['position']+staff_lines[idx[5]]['position'])/2
    else:
        if hand=='left':
            if idx[0]==0:
                six = staff_lines[idx[0]]['position'] - offset
            else:
                six = (staff_lines[idx[0]]['position']+staff_lines[idx[0]-1]['position'])/2
            eix = (staff_lines[idx[4]]['position']+staff_lines[idx[5]]['position'])/2
        elif 'hand'=='right':
            six = (staff_lines[idx[5]]['position']+staff_lines[idx[4]]['position'])/2
            if idx[10]==len(staff_lines):
                eix = staff_lines[idx[10]]['position'] + offset
            else:
                eix = (staff_lines[idx[10]]['position']+staff_lines[idx[10]+1]['position'])/2
    
    return sheet[int(six):int(eix),:]


def generate_images(numimages = 10, maxclasses = 10, imsize = (320, 160), numboxes = (35, 17), staff_size = 20, 
                    datadir = 'C:\\Users\\Niki\\Source\\SmartSheetOMR\\data'):
    
    classdir = 'C:\\Users\\Niki\\Datasets\\Music Symbols\\Handmade\\Objects'
    classes = [name for name in os.listdir(classdir) if os.path.isdir(os.path.join(classdir, name))]
  
    heights = {}
    with open(os.path.join(classdir, 'heights.csv')) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            k,v = row
            heights[k] = float(v)
    
    annotations = []
    for i in range(numimages):
        image = Image.new("L", imsize, "white")
        imname = str(i)+'.png'
        ann = [imname]
        space = int(staff_size/2 + staff_size*np.random.rand())
        notes = np.array([6,18])
        nc = np.random.randint(0, len(classes), np.random.randint(3, maxclasses*min([1,(staff_size/space)**2])))
        nc = np.concatenate((nc, notes))
        
        for c in nc:
            thispath = os.path.join(classdir, classes[c])
            samp = os.listdir(thispath)
            sampix = np.random.randint(0, len(samp))
            im = Image.open(os.path.join(thispath, samp[sampix]))
            
            w,h = im.size
            newh = (space+1)*heights[classes[c]]
            
            ratio = newh/h
            x = int(w*ratio)
            y = int(newh)
            im = im.resize((x,y))
    
            sx = np.random.randint(0, imsize[0]-x)
            sy = np.random.randint(0, imsize[1]-y)
    
            image.paste(im, (sx,sy)) #, mask=im
            ann.append(sx)
            ann.append(sy)
            ann.append(sx+x)
            ann.append(sy+y)
            ann.append(c)
            
        annotations.append(ann)
        
        # Add Lines
        draw = ImageDraw.Draw(image)
        start = np.random.randint(0, image.size[1]-10*space)
        lw = int(1+np.round(np.random.rand()))
        for l in range(0,10):
            draw.line((0, start+space*l, image.size[0], start+space*l), width = lw)    
        
        image.save(os.path.join(datadir, 'images', imname))
        
    # Save annotaions
    cut = int(0.9*len(annotations))
    with open(os.path.join(datadir, 'annotations_train.csv'), "w") as f:
        writer = csv.writer(f, lineterminator = '\n', delimiter = ' ')
        writer.writerows(annotations[:cut])
            
    with open(os.path.join(datadir, 'annotations_val.csv'), "w") as f:
        writer = csv.writer(f, lineterminator = '\n', delimiter = ' ')
        writer.writerows(annotations[cut:])

    # Save classes
    f = open(os.path.join(datadir, 'classes.csv'), 'w')
    for cl in classes: 
        f.write("%s\n" % cl)
    f.close()    
 
def load_data(numboxes=(35, 17), imsize = (320, 160), datadir = 'C:/Users/Niki/Source/SmartSheetOMR/data', locations = True):
    # Load Targets    
    classes = csv_to_list(os.path.join(datadir, 'classes.csv'))
    numclasses = len(classes)
    imdir = os.path.join(datadir, 'images')
    Y_train = []
    X_train = []    
    rows = csv_to_list(os.path.join(datadir, 'annotations_train.csv'), single = False)
    for row in rows:
        if locations:
            labels = annotations_to_locations(numboxes, numclasses, imsize, row)
        else:
            labels = annotations_to_labels(numboxes, numclasses, imsize, row)
        Y_train.append(labels)
        # Load corresponding image
        im = Image.open(os.path.join(imdir, row[0]))
        im = np.array(im).astype('float')/255
        X_train.append(im)        

    Y_val = []
    X_val = []    
    rows = csv_to_list(os.path.join(datadir, 'annotations_val.csv'), single = False)
    for row in rows:
        if locations:
            labels = annotations_to_locations(numboxes, numclasses, imsize, row)
        else:
            labels = annotations_to_labels(numboxes, numclasses, imsize, row)
        Y_val.append(labels)
        # Load corresponding image
        im = Image.open(os.path.join(imdir, row[0]))
        im = np.array(im).astype('float')/255
        X_val.append(im)      
        
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_val), np.asarray(Y_val)

def annotations_to_labels(numboxes,numclasses,imsize,row):
    dim = (numclasses+3)*numboxes[0]*numboxes[1]
    z = np.zeros(dim)
    for p in range(1, len(row), 5):
        sx,sy,ex,ey,c = row[p],row[p+1],row[p+2],row[p+3],row[p+4]

        cx = (float(sx)+float(ex))/2
        cy = (float(sy)+float(ey))/2
        
        x = int(cx/imsize[0]*numboxes[0])
        y = int(cy/imsize[1]*numboxes[1])        
        
        idx = (numclasses+3)*(x*numboxes[1]+y)
        z[idx] = 1
        z[idx+1] = x/numboxes[0]
        z[idx+2] = y/numboxes[1]
        z[idx+3+int(c)] = 1
  
    return z
    
def annotations_to_locations(numboxes,numclasses,imsize,row):
    dim = 5*numboxes[0]*numboxes[1]
    box_size = [imsize[0]/numboxes[0], imsize[1]/numboxes[1]]
    z = np.zeros(dim)
    for p in range(1, len(row), 5):
        sx,sy,ex,ey = row[p],row[p+1],row[p+2],row[p+3]

        cx = (float(sx)+float(ex))/2
        cy = (float(sy)+float(ey))/2
        
        bx = int(cx/box_size[0])
        by = int(cy/box_size[1])
        idx = 5*(bx*numboxes[1]+by)
        
        rx = cx/box_size[0]-bx
        ry = cy/box_size[1]-by
        
        dx = (float(ex)-float(sx))/box_size[0]
        dy = (float(ey)-float(sy))/box_size[1]
               
        z[idx] = 1
        z[idx+1] = rx
        z[idx+2] = ry
        z[idx+3] = dx
        z[idx+4] = dy
    return z

class Net(nn.Module):
    def __init__(self, outdim=10, middim = 200):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(4),
            nn.SELU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.SELU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.SELU(),
            nn.MaxPool2d(2))  
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.MaxPool2d(2))              
        self.flat = nn.Sequential(
            nn.Linear(64*10*5, middim),
            nn.BatchNorm2d(middim),
            nn.SELU())
        self.final = nn.Linear(middim, outdim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        #print(out.shape)        
        out = out.view(out.size(0), -1)
        out = self.flat(out)
        out = F.dropout(out, p=0.3)
        out = F.selu(self.final(out))
        return out

def train(model, optimizer, train_data, val_data, criterion, epochs=5):   
    batch = 90
    nsamples = train_data[1].shape[0]
    train_loss = []
    val_loss = []
    best_loss = np.inf
    best_model = model
    for epoch in range(epochs):  # loop over the dataset multiple times    
        running_loss = 0.0
        model.train(True)
        ctr = 0
        for b in range(0, nsamples, batch):
            ctr+=1
            # get the inputs
            s = slice(b,b+batch)
            inputs, labels = train_data[0][s,:,:], train_data[1][s,:]

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.data[0]
            #if b % 2 == 1:    # print every 10 mini-batches
            print("Epoch: {};  Batch: {}; Loss: {}".format(epoch + 1, int(b/batch) + 1, loss.data[0]))
            
        train_loss.append(running_loss/ctr)
        
        # Compute Validation Loss
        model.train(False)
        inputs, labels = Variable(val_data[0]), Variable(val_data[1])
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss.append(loss.data[0])
        print("Epoch: {};  Average Loss: {}; Validation Loss: {}".format(epoch + 1, train_loss[-1], val_loss[-1]))
        
        if val_loss[-1]<best_loss:
            best_loss = val_loss[-1]
            best_model = model
            
        checkdata = {
            'epoch': epoch + 1,
            'state_dict': best_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }            
        torch.save(checkdata, 'checkpoint.pt')
        
    print('Finished Training')
    return model
   
def eval_locations(model, sample, target = None, numboxes=(35, 17), datadir = 'C:\\Users\\Niki\\Source\\SmartSheetOMR\\data'):
    model.train(False)
    if target is None:
        v = Variable(sample)
        pred = model(v)
        pred = pred.data.numpy()
    else:
        pred = target
    
    boxix = np.arange(numboxes[0]*numboxes[1])*5 
    detections = boxix[np.where(pred[0,boxix]>0.5)[0]]
    
    im = Image.fromarray(np.uint8(sample.numpy()[0,0,:,:])*255)
    draw = ImageDraw.Draw(im)
    imx,imy = im.size
    box_size = [imx/numboxes[0], imy/numboxes[1]]
    rec = []
    for d in detections:
        rx,ry,dx,dy= pred[0,d+1], pred[0,d+2], pred[0,d+3], pred[0, d+4]
        
        bx = box_size[0]*int(d/numboxes[1]/5)
        by = box_size[1]*(int(d/5) % numboxes[1])
        
        dx = dx*box_size[0]
        dy = dy*box_size[1]

        sx = bx + rx*box_size[0] - dx/2
        sy = by + ry*box_size[1] - dy/2
        ex = bx + rx*box_size[0] + dx/2
        ey = by + ry*box_size[1] + dy/2
        
        rec.append([int(sx), int(sy), int(ex), int(ey)])        
        draw.rectangle([rec[-1][0], rec[-1][1], rec[-1][2], rec[-1][3]])
        
    im.show()
    return np.asarray(rec), pred

    
def eval_all(model, sample, target = None, numboxes=(35, 17), datadir = 'C:\\Users\\Niki\\Source\\SmartSheetOMR\\data'):
    classfile = os.path.join(datadir, 'classes.csv')
    classes = csv_to_list(classfile)
    nclasses = len(classes)
    if target is None:
        v = Variable(sample)
        pred = model(v)
        pred = pred.data.numpy()
    else:
        pred = target
    
    boxix = np.arange(numboxes[0]*numboxes[1])*(nclasses+3)   
    detections = boxix[np.where(pred[0,boxix]>0.5)[0]]
    
    im = Image.fromarray(np.uint8(sample.numpy()[0,0,:,:])*255)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 10)
    w,h = im.size
    x = []
    y = []
    c = []
    for d in detections:
        vals = pred[0, d:d+nclasses+3]
        x.append(vals[1]) 
        y.append(vals[2])
        c.append(classes[vals[3:].argmax()])
        draw.text((int(x[-1]*w), int(y[-1]*h)),c[-1],font=font)
    im.show()
    return x,y,c, pred
    
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def csv_to_list(csvfile, single = True):
    with open(csvfile, "r") as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            if single:
                rows.append(row[0])
            else:
                rows.append(row)
    return rows
