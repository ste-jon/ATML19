import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import os
from itertools import chain
from torchvision import utils


"""
This WGAN-GP Model was obtained from: github.com/Zeleni9/pytorch-wgan/,
a collection of various GAN models used for CIFAR, Fashion MNIST.

Adjustments were made to match our custom parsing arguments,
match our image sizes, wished learning rate, convolutional filters/complexity,
training output, image saving and speed of training.

Several extras were ommitted, such as TensorBoard logging and Inception score calculation.

Note: Some torch vectors need to be defined differently per version, i.e. the one/mone vector.
"""

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # Z latent vector 128
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # (512x4x4)
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            # (4x8x8)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            # (4x16x16)
            nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # => (4x32x32)
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Output_dim = 1
        self.main_module = nn.Sequential(

            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of D --> (4x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class WGAN_GP(object):
    def __init__(self, args):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 2e-4  # set higher
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10


    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))


    def train(self, train_loader):

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1.0)
        mone = torch.tensor(-1.0)
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 128, 1, 1))

                if self.cuda:
                    images, z = Variable(images.cuda(self.cuda_index)), Variable(z.cuda(self.cuda_index))
                else:
                    images, z = Variable(images), Variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 128, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 128, 1, 1))
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward(retain_graph=True)


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = Variable(torch.randn(self.batch_size, 128, 1, 1)).cuda(self.cuda_index)
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            # Saving model and sampling images every 100th generator iterations
            if (g_iter) % 100 == 0:

                # Denormalize images and save them in grid 4x4
                z = Variable(torch.randn(800, 128, 1, 1)).cuda(self.cuda_index)
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:16]
                grid = utils.make_grid(samples)
                utils.save_image(grid, 'GAN/img_iter_{}.png'.format(str(g_iter).zfill(3)), 4)

                # Print training state for supervision
                print("Generator iter: {}".format(g_iter))

                dl = '{:3f}'.format(d_loss.data.item())
                gc = '{:3f}'.format(g_cost.data.item())
                print("Loss D " + dl + " Generator Cost: " + gc)


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])


    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(self.C, 32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images