from WGAN_GP import WGAN_GP
import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import ImageFolder

# parser to set dynamic options
parser = argparse.ArgumentParser(description='Script to run the Wasserstein GP GAN')
parser.add_argument("--n_epochs", type=int, default=10, help="epochs for GAN training")
parser.add_argument('--batch_size', default=64, help='batch size')
parser.add_argument('--dataset_path', '-p', default='/var/tmp/covers/GAN', help='Folder with the category images to create images from')
parser.add_argument('--cuda', default=True, help='Activate CUDA')
parser.add_argument("--img_size", type=int, default=32, help="image dimensions")
parser.add_argument("--is_train", default=True, help="train mode")
parser.add_argument("--channels", type=int, default=3, help="train mode")
parser.add_argument("--generator_iters", type=int, default=2000, help="no. of iterations")

args = parser.parse_args()
print(args)

def main(args):

    # load the wished dataset (WGAN-GP: We only train it on a subcategory, manually moved)
    dataset_path = args.dataset_path
    target_size = (args.img_size, args.img_size)
    transforms = Compose([Resize(target_size),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

    train_dataset = ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(dataloader.dataset))

    # initiative the model, relevant args: Channels, BatchSize, iterations, cuda, train
    model = WGAN_GP(args)

    # Load datasets to train and test loaders
    # Start model training
    if args.is_train is True:
        model.train(dataloader)



if __name__ == '__main__':
    main(args)