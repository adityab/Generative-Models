import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from a Generative Model.')
    parser.add_argument('path', help='Model file path (.h5)')
    args = parser.parse_args()

    model = MNIST_DCGAN(z_dim=100)
    model.load_saved(args.path)
    samples = sample_normal(model, 100)
    save_images(args.path.split('.')[0]+'.png', samples, 10, 10)
