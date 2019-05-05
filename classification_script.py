import math
import numpy as np
import os
import matplotlib.pyplot as plt


# import the covers
cover_dir = 'data/covers'
cover_paths = os.listdir(cover_dir)
cover_paths.sort()
print(cover_paths[:10])

from PIL import Image
from utils_labels import category_string_to_number

# converts the jpg into numpy arrays of (target H, W)
# additionally extracts the label = category from the filename

# TODO: might be slow for 100k+ images, different possibilities?
def load_images(data_root, image_paths, target_size):
    images = []
    labels = []
    for img_path in image_paths:
        path = os.path.join(data_root, img_path)
        img = Image.open(path)
        img = img.resize(target_size)
        
        img_label = img_path[img_path.find("_")+1:-4]
        img_label = category_string_to_number(img_label)

        if(img_label >= 0):
            images.append(np.array(img))
            labels.append(img_label)
    return images, labels

n_imgs = 4000
# defined by the imported pytorch models, requires 224+x224+ (ImageNet)
target_size = (224, 224)



### Dataset preparation:
cover_images, cover_labels = load_images(cover_dir, cover_paths[:n_imgs], target_size)

# shuffle
np.random.shuffle(cover_images)

# set aside 10% for testing
n_test = math.floor(len(cover_images) * 0.10)
covers_test = cover_images[:n_test]
labels_test =  cover_labels[:n_test]

# split in test and validation set.
# given in percentage of samples except test set.
val_split = 0.2
n_val = n_test + math.floor( (len(cover_images)-n_test) * val_split)
covers_val = cover_images[n_test:n_val]
labels_val =  cover_labels[n_test:n_val]

covers_train = cover_images[n_val:]
labels_train =  cover_labels[n_val:]

# Normalization here given by pytorch (pretrained Alexnet -- Densenet?)
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
transform = Compose([   #Resize(224,224),
                        ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                    ])

from torch.utils.data import Dataset

class CoverDataset(Dataset):
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform=transform
    def __getitem__(self, index):
        image = transform(self.data[index])
        target = torch.tensor(self.target[index]).long()
        return image, target
    def __len__(self):
        return len(self.data)
    
    
# define the datasets
train_dataset = CoverDataset(covers_train, labels_train, transform)
val_dataset = CoverDataset(covers_val, labels_val, transform)
test_dataset = CoverDataset(covers_test, labels_test, transform)


print("Amount in the training set: " + str(len(train_dataset)))
print("Amount in the validation/test set: " + str(len(test_dataset)))


# define the dataloaders, batch_size 128, 64, 32? Needs to be adjusted for the cluster
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# torch preparation
import torch
from torchvision import *
import torchvision.models as models
import torch.nn as nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import torch.nn as nn
from utils_train import train, test, fit



### 
# 1) Import a pretrained Alexnet model from pytroch
# 2) Fix the weights for each layers
# 3) replace the last output layer with our custom one
###

num_classes = 30 # might change to 32?

alexnet = models.alexnet(pretrained=True)


for param in alexnet.parameters():
    param.requires_grad = False # == do not change weights, do not re-train


## fixed, pre-trained alexnet. Now, replace the last layer:
alexnet.classifier._modules['6'] = nn.Linear(4096, num_classes)
print(*list(alexnet.children()))  # show the model (optional)


# needs to be defined on the cluster training procedure
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.003)


alexnet = alexnet.to(device)
n_epochs = 1
# retrain (only that last replaced layer)
alexnet_retrain = fit(train_loader, val_loader, model=alexnet, optimizer=optimizer, loss_fn=loss_fn, n_epochs=n_epochs)
### 
# same procedure with "densenet161" (good performance on ImageNet, new concept)
###

densenet = models.densenet161(pretrained=True)

# fix the weights
for param in densenet.parameters():
    param.requires_grad = False

# replace the last output layer
densenet.classifier =  nn.Linear(2208, num_classes)
print(*list(densenet.children()))

# retrain densenet
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.003)
densenet = densenet.to(device)

n_epochs = 1
densenet_retrain = fit(train_loader, val_loader, densenet, optimizer, loss_fn, n_epochs)
