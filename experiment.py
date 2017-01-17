import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import argparse

from models.mnist_dcgan import MNIST_DCGAN

def sample_normal(model, n_samples):
    # Extract samples from the model
    latents = np.random.normal(0, 1, size=(n_samples, model.z_dim))
    return model.predict(latents)

def save_images(path, samples, rows, cols):
    # Save sampled images in a grid
    f, ax = plt.subplots(rows, cols)
    k = 0
    for i in range(rows):
        for j in range(cols):
            img = samples[k][0]
            ax[i, j].imshow(img, interpolation='none')
            ax[i, j].axis('off')
            k = k + 1

    f.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(f)

def train(args):
    start_time = None

    if not os.path.exists('results'):
        os.mkdir('results')

    if args.model == 'mnist_dcgan':
        model = MNIST_DCGAN(z_dim=100)

        def before_epoch(epoch):
            print('Epoch %d ...' % epoch)
            start_time = time.time()

        def after_epoch(epoch, loss_gan, loss_discriminator):
            print('GAN loss: \t', loss_gan)
            print('Discriminator loss: \t', loss_discriminator)
            samples = sample_normal(model.generator, 100)
            model.generator.save('results/mnist_dcgan-z' + model.z_dim + '-epoch' + epoch + '.h5')
            save_images('results/mnist_dcgan-z' + model.z_dim + '-epoch' + epoch + '.png', samples, 10, 10)

        model.train(args.batch_size, args.epochs, before_epoch, after_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Generative Model.')

    parser.add_argument('model', help='Model name')
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', help='Batch Size')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', help='Number of epochs')

    args = parser.parse_args()
    train(args)
