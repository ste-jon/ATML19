import matplotlib.pyplot as plt
import numpy as np
import re
from torchvision.datasets import ImageFolder
from utils_labels import idx_to_class, folder_to_cat_dict


def load_per_label(savedir, dataset_name, subset):
    dataset = ImageFolder('data/covers/test')
    folder_to_category = folder_to_cat_dict('data/meta/folder_to_cat.json')
    class_to_idx = dataset.class_to_idx

    out = []
    with open(savedir + dataset_name +  '_' + subset +'_results.txt', 'r') as file:
        data = file.readlines()
        data = data[3:]
        for line in data:
            numbers = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line)
            tmp = numbers[3]
            numbers = numbers[0:2]
            numbers.append(tmp)
            numbers[0] = folder_to_category[idx_to_class(int(numbers[0]), class_to_idx)]
            out.append(numbers)

    return  np.asarray(out)


def load_learning(savedir, dataset_name, subset):
    out = []
    with open(savedir + dataset_name +  '_' + subset +'_learning_iterations.txt', 'r') as file:
        data = file.readlines()
        data = data[1:]
        for line in data:
            numbers = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line)
            out.append(numbers)

    return  np.asarray(out)


def load_acc(savedir, dataset_name, subset):
    with open(savedir + dataset_name +  '_' + subset +'_results.txt', 'r') as file:
        data = file.readlines()
        top1 = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', data[1])[0]
        top3 = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', data[2])[1]

    return  top1, top3


def plot(labels, data1, label1, data2, label2, filename = None):
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35  # the width of the bars

    print(ind)
    print(labels)
    print(data1)

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 =ax.bar(ind, data1, width, bottom = 0,
                    color='SkyBlue')
    #rects1 =ax.bar(ind - width/2, data1, width, bottom = 0,
    #                color='SkyBlue', label=label1)
    #rects2 =ax.bar(ind + width/2, data2, width, bottom = 0,
    #                color='IndianRed', label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('accuracy')
    ax.set_title('Comparison between normalized and non normalized dataset trained on Alexnet')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    if filename is not None:
        plt.savefig('plots/'+filename, dpi = 150)


def plot_learning(data, filename = None):
    fig, ax = plt.subplots(figsize=(30, 20))
    x = np.arange(len(data[:,0]))
    
    line1, = ax.plot(x, data[:,0], label='Loss')
    line1.set_dashes([2, 2, 10, 2]) 

    line2, = ax.plot(x, data[:,1], label='Top1 Accuracy')
    
    line3, = ax.plot(x, data[:,2], label='Top3 Accuracy')

    ax.legend()
    if filename is not None:
        plt.savefig('plots/'+filename, dpi = 150)


# strictly for GAN postprocessed csv files only
import csv
def plot_gan_losses(savedir, csv_name, filename):
    epochs = []
    d_losses = []
    g_losses = []
    lossreader = csv.reader(open(savedir + csv_name, newline=''), delimiter=';')
    # skip header:
    next(lossreader)
    for row in lossreader:
        epochs.append(int(row[0]))
        d_losses.append(float(row[1]))
        g_losses.append(float(row[2]))

    # plot the collected data
    plt.plot(epochs, d_losses, color='r', label='D Loss')
    plt.plot(epochs, g_losses, color='g', label='G Loss')
    #plt.axis([0, 500, -10, 10])
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN GP loss')
    if filename is not None:
        plt.savefig('plots/'+filename, dpi = 150)



