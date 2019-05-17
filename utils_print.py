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


def plot(labels, data1, label1, data2, label2):
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, data1, width,
                    color='SkyBlue', label=label1)
    rects2 = ax.bar(ind + width/2, data2, width,
                    color='IndianRed', label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('norm / non norm comparison')
    ax.set_title('Comparison between normalized and non normalized dataset trained on Alexnet')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend()


def plot_learning(data):
    fig, ax = plt.subplots()
    x = np.arange(len(data[:,0]))
    
    line1, = ax.plot(x, data[:,0], label='Loss')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    line2, = ax.plot(x, data[:,1], label='Top1 Accuracy')
    
    line3, = ax.plot(x, data[:,2], label='Top3 Accuracy')

    ax.legend()