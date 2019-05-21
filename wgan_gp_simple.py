import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import ImageFolder


"""
The main WGAN GP architecture was obtained from github.com/eriklindernoren/PyTorch-GAN,
a collection of GANs for MNIST digits production. 
(github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)

Several adjustments regarding image type (channels, size), architecture, learning rate,
output for supervision, image saving, paths etc.
were done to fit the architecture to our task.

Please not that manual directory creation might be necessary (cluster).
"""

# for image saving on the cluster
os.makedirs("gan/fakes/simple", exist_ok=True)

# parser to set dynamic options
parser = argparse.ArgumentParser(description='Script to run the Wasserstein GP GAN')
parser.add_argument("--n_epochs", type=int, default=100, help="epochs for GAN training")
parser.add_argument('--batch-size', '-b', default=64, help='batch size')
parser.add_argument('--dataset_path', '-p', default='/var/tmp/covers/GAN', help='Folder with the category images to create images from')
parser.add_argument("--img_size", type=int, default=28, help="image dimensions")
parser.add_argument("--lr", type=float, default=0.0002, help="GAN learning rate")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
# from the referenced Github implementation:
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=800, help="interval betwen image samples")

args = parser.parse_args()
print(args)


cuda = True if torch.cuda.is_available() else False
img_shape = (3, args.img_size, args.img_size)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 768),
            *block(768, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(768, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
# import the covers
dataset_path = args.dataset_path

target_size = (args.img_size, args.img_size)
transforms = Compose([Resize(target_size),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

train_dataset = ImageFolder(dataset_path, transform=transforms)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()


            if batches_done % args.sample_interval == 0:
                save_image(fake_imgs.data[:16], "GAN/fakes/simple/%d.png" % batches_done, nrow=4, normalize=True)

            batches_done += args.n_critic

        
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, args.n_epochs, d_loss.item(), g_loss.item())
    )