# Example usage:
# python train.py /home/workspace/aipnd-project/flowers --save_dir '/home/workspace/aipnd-project/' -a vgg19 -u 1050 -l 0.001 --gpu -e 1

import sys
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('data_dir', help = 'Data directory')
parser.add_argument('-s', '--save_dir', help = 'Define save directory for checkpoint')
parser.add_argument('-a', '--arch', help = 'Architecture (e.g: VGG13)')
parser.add_argument('-l', '--learning_rate', required = True, type=float, help = 'Learning Rate')
parser.add_argument('-u', '--hidden_units', required = True, type=int, help = 'Hidden Units')
parser.add_argument('-e', '--epochs', required = True, type=int, help = 'Epochs')
parser.add_argument('--gpu', action='store_true', help = 'Enable GPU')

args = parser.parse_args()

# data_dir = '/home/workspace/aipnd-project/flowers'
data_dir = args.data_dir
train_dir = args.data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
save_dir = args.save_dir

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 32)

def choose_architecture(name):
    if name == 'vgg11':
        return models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        return models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        return models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        return models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        return models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        return models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        return models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        return models.vgg19_bn(pretrained = True)
    else: # Error message
        print("This project supports only VGG models. You can choose one of the following: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn")
        sys.exit()
        
model = choose_architecture(args.arch)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, args.hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dp2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))     
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
        
cudaEnabled = True if args.gpu else False

if cudaEnabled:
    model.to('cuda')

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        
        if cudaEnabled:
            images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy    
    
epochs = args.epochs
print_every = 50
running_loss = 0
steps = 0

for e in range(epochs):
    model.train()
    for images, labels in iter(train_loader):
         steps += 1
         if cudaEnabled:
            images, labels = images.to('cuda'), labels.to('cuda')
         optimizer.zero_grad()
         output = model.forward(images)
         loss = criterion(output, labels)
         loss.backward()
         optimizer.step()
        
         running_loss += loss.item()
         if steps % print_every == 0:
            model.eval()
            
            with torch.no_grad():
                test_loss, accuracy = validation(model, valid_loader, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
    
            running_loss = 0
            model.train()
        
checkpoint = {
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'classifier': model.classifier
}

torch.save(checkpoint, save_dir + '/checkpoint.pth')
        
if __name__ == '__main__':
    print("Checkpoint saved in: {}".format(save_dir))
    