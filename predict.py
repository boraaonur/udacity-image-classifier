# Example usage:
# python predict.py -c /home/workspace/aipnd-project --gpu --top_k 3 /home/workspace/aipnd-project/flowers/test/100/image_07897.jpg --category_names cat_to_name.json

from PIL import Image
import argparse
import torch
import numpy as np
from torch.autograd import Variable
import json
from torchvision import models

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('filepath', help = 'Filepath to image')
parser.add_argument('-c', '--checkpoint_path', help = 'Filepath to load checkpoint')
parser.add_argument('--gpu', action='store_true', help = 'Enable GPU')
parser.add_argument('--category_names', help = 'Select JSON file')
parser.add_argument('--top_k', type = int, help = 'Define Top-K')
args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained = True)
    elif checkpoint['arch'] == 'vgg11_bn':
        model = models.vgg11_bn(pretrained = True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif checkpoint['arch'] == 'vgg13_bn':
        model = models.vgg13_bn(pretrained = True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif checkpoint['arch'] == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif checkpoint['arch'] == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

model = load_checkpoint(args.checkpoint_path + '/checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize Image
    im = Image.open(image)
    width, height = im.size
    ratio = width / height
    im = im.resize((256, int(ratio * 256)))

    # Crop Image
    left = (im.size[0] - 224)/2
    top = (im.size[1] - 224)/2
    right = (im.size[0] + 224)/2
    bottom = (im.size[1] + 224)/2
    im = im.crop((left, top, right, bottom))
    
    # Color stuff
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 1, 0))
    return np_image

cudaEnabled = True if args.gpu else False

if cudaEnabled:
    model.to('cuda')

def predict(image_path, model):
    with torch.no_grad():
        model.eval()
        
        topk_enabled = False if args.top_k == None else True
        
        np_image = process_image(image_path)
        image_tensor = torch.from_numpy(np_image)

        inputs_var = Variable(image_tensor.float())    
        
        if cudaEnabled:
            model.to('cuda')
            inputs_var = Variable(image_tensor.float().cuda()) 
        else:
            model.to('cpu')
            inputs_var = Variable(image_tensor.float()) 
                
        output = model.forward(inputs_var.unsqueeze(0))
        ps = torch.exp(output).topk(1)
        
        if topk_enabled:
            ps = torch.exp(output).topk(args.top_k)
        
        probs = ps[0].cpu()
        classes = ps[1].cpu()
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[each] for each in classes.numpy()[0]]
        
        name_list = []
        with open(args.category_names) as file:
            data = json.load(file)
            for flower_id in top_classes:
                 name_list.append(data[str(flower_id)])
                    
        if topk_enabled:
            print("Top {} probabilities: ".format(args.top_k) + "%")
            print(probs.numpy()[0] * 100)
            print("Top {} classes: ".format(args.top_k))
            print(name_list)
        probability = round(probs.numpy()[0].tolist()[0] * 100, 2)
        flower_name = name_list[0].title()
        print("Flower name is: {} with {}% probability".format(flower_name, probability))
    return probs.numpy()[0].tolist(), name_list

predict(args.filepath, model)