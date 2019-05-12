import math
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader
from utils_labels import idx_to_class
from utils_labels import folder_to_cat_dict
from utils_save import save_performance, save_test_results

import torch
from torchvision import *
import torchvision.models as models
import torch.nn as nn
from utils_train import train, test, fit


class Classification:

    def __init__(self, data_path, meta_path, batch_size, save_dir = ''):
        # torch preparation
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # import the covers
        cover_dir_train = data_path+'train'
        cover_dir_test = data_path+'test'
        cover_dir_val = data_path+'valid'

        # Normalization here given by pytorch (pretrained Alexnet -- Densenet?)
        target_size = (224, 224)
        transforms = Compose([Resize(target_size),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        train_dataset = ImageFolder(cover_dir_train, transform=transforms)
        test_dataset = ImageFolder(cover_dir_test, transform=transforms)
        val_dataset = ImageFolder(cover_dir_val, transform=transforms)

        # class is the folder name and idx a number generated by ImageFolder
        self.class_to_idx = train_dataset.class_to_idx
        # mapping of folder to category name
        folder_cat_path = meta_path+'folder_to_cat.json'
        self.folder_to_category = folder_to_cat_dict(folder_cat_path)

        self.num_classes = len(self.class_to_idx)

        print("Amount in the training set: " + str(len(train_dataset)))
        print("Amount in the test set: " + str(len(test_dataset)))
        print("Amount in the validation set: " + str(len(test_dataset)))

        # define the dataloaders, batch_size 128, 64, 32? Needs to be adjusted for the cluster
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.savedir = save_dir
        self.loaddir = save_dir
    

    def train(self, network_name='alexnet', n_epochs = 1, do_print = True):

        if 'alexnet' == network_name:
            model, optimizer, loss_fn = self.setup_alexnet()

        if 'densenet' == network_name:
            model, optimizer, loss_fn = self.setup_densenet()

        performance = fit(self.train_loader, self.val_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, n_epochs=n_epochs)

        # training returns:
        best_model = performance[8] # the model saved by early stopping in the fit procedure
        val_loss, val_acc, val_3_acc = performance[3][-1], performance[4][-1], performance[5][-1]
        accuracy_per_label, accuracy_per_label_top3 = performance[6:8]

        if do_print:
            # Print validation accuracy
            print('Validation loss: {:.4f}, Validation accuracy: {:.4f}, Validation top3 accuracy: {:.4f}'.format(val_loss, 
                                                                                                                    val_acc, 
                                                                                                                    val_3_acc))
            
            # Print per lable accuracy
            for it, acc in enumerate(accuracy_per_label):
                    print('{}: val_accuracy: {:.4f}, top 3 val_accuracy: {:.4f}'.format(self.folder_to_category[idx_to_class(it, self.class_to_idx)], accuracy_per_label[it], accuracy_per_label_top3[it]))

        if self.savedir != '':
            try:
                os.mkdir(self.savedir)
                print("Directory " , self.savedir ,  "Save directory Created ") 
            except FileExistsError:
                print("Directory " , self.savedir ,  "Save directory already exists")
            torch.save(best_model, self.savedir + network_name +'_model_parameter.pt')
            save_performance(self.savedir, network_name + '_train', performance[0:3])
            save_performance(self.savedir, network_name + '_validation', performance[3:8])

    ###
    # 1) Import a pretrained Alexnet model from pytroch
    # 2) Fix the weights for each layers
    # 3) replace the last output layer with our custom one
    ###
    def setup_alexnet(self):
        
        alexnet = models.alexnet(pretrained=True)
        
        for param in alexnet.parameters():
            param.requires_grad = False  # == do not change weights, do not re-train

        ## fixed, pre-trained alexnet. Now, replace the last layer:
        alexnet.classifier._modules['6'] = nn.Linear(4096, self.num_classes)
        print(*list(alexnet.children()))  # show the model (optional)

        # needs to be defined on the cluster training procedure
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.001)
        alexnet = alexnet.to(self.device)

        return alexnet, optimizer, loss_fn

    ###
    # 1) Import a pretrained densenet161 model from pytroch (good performance on ImageNet, new concept)
    # 2) Fix the weights for each layers
    # 3) replace the last output layer with our custom one
    ###   
    def setup_densenet(self):
        densenet = models.densenet161(pretrained=True)

        # fix the weights
        for param in densenet.parameters():
            param.requires_grad = False

        # replace the last output layer
        densenet.classifier =  nn.Linear(2208, self.num_classes)
        print(*list(densenet.children()))

        # retrain densenet
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)
        densenet = densenet.to(self.device)

        return densenet, optimizer, loss_fn


    def test(self, network_name = 'alexnet'):

        model = torch.load(self.loaddir + network_name + '_model_parameter.pt')
        loss_fn = nn.CrossEntropyLoss()
        test_output = test(model, self.test_loader, loss_fn)

        # Print test accuracy
        print('Test loss: {:.4f}, Test accuracy: {:.4f}, top3 accuracy: {:.4f}'.format(test_output[0],
                                                                                    test_output[1],
                                                                                    test_output[2]))
        # Print per lable accuracy
        accuracy_per_label, accuracy_per_label_top3 = test_output[3:5]
        for it, acc in enumerate(accuracy_per_label):
                print('{}: accuracy: {:.4f}, top 3 accuracy: {:.4f}'.format(self.folder_to_category[idx_to_class(it, self.class_to_idx)], accuracy_per_label[it], accuracy_per_label_top3[it]))

        if self.savedir != '':
            save_test_results(self.savedir, network_name + '_test', test_output)