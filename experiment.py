import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import time
import argparse

from models.mnist_dcgan import MNIST_DCGAN

def sample_normal(model, n_samples):
    # Extract samples from the model
    latents = np.random.normal(0, 1, size=(n_samples, 100))
    return model.generate(latents)

def save_images(path, samples, rows, cols):
    # Save sampled images in a grid
    f = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.0, hspace=0.0)
    k = 0
    for i in range(rows):
        for j in range(cols):
            img = samples[k]
            ax = plt.subplot(gs[k])
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            k = k + 1

    f.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(f)

def train(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model == 'mnist_dcgan':
        model = MNIST_DCGAN(z_dim=100)

        def before_epoch(epoch):
            print('Epoch %d ...' % epoch)

        def after_epoch(epoch, loss_gan, loss_discriminator):
            print('\n')
            print('GAN loss: \t', loss_gan)
            print('Discriminator loss: \t', loss_discriminator)
            samples = sample_normal(model, 100)
            model.generator.save(args.output_dir + '/mnist_dcgan-z' + str(model.z_dim) + '-epoch' + str(epoch) + '.h5')
            save_images(args.output_dir + '/mnist_dcgan-z' + str(model.z_dim) + '-epoch' + str(epoch) + '.png', samples, 10, 10)

        model.train(args.batch_size, args.epochs, before_epoch, after_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Generative Model.')

    parser.add_argument('model', help='Model name')
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', help='Batch Size')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', help='Number of epochs')
    parser.add_argument('-o', '--output_dir', type=str, dest='output_dir', help='Output Directory')
    args = parser.parse_args()
    train(args)
