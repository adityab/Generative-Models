import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.optimizers import SGD, Adam
from keras.datasets import mnist

class MNIST_DCGAN:
    def __init__(self, z_dim):
        self.input_dim = (None, 28, 28)
        self.z_dim = 100
        self.n_filters = 64
        self.generator = self._generator()

    def _generator(self):
        net = Sequential([
            Dense(1024, input_dim=self.z_dim, init='glorot_normal'),
            BatchNormalization(),
            LeakyReLU(alpha=0.3)
            Reshape(size=(self.n_filters * 2,7,7)),
            Deconvolution2D(self.n_filters, 5, 5, subsample=(2,2), border_mode='same')
            BatchNormalization(),
            LeakyReLU(alpha=0.3)
            Deconvolution2D(1, 5, 5, subsample=(2,2), border_mode='same')
            Activation('tanh')
        ])

        latent = Input(shape=(self.z_dim,))
        image = net(latent)

        return Model(input=latent, output=image)

    def _discriminator(self):
        net = Sequential([
            Convolution2D(self.n_filters, 5, 5, subsample=(2,2), border_mode='same', init='glorot_normal')
            LeakyReLU(alpha=0.2)
            Convolution2D(self.n_filters * 2, 5, 5, subsample=(2,2), border_mode='same', init='glorot_normal')
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Flatten()
            Dense(1)
            Activation('sigmoid')
        ])

        image = Input(shape=(1, 28, 28))
        prediction = net(image)

        return Model(input=image, output=prediction)

    def _GAN(self, generator, discriminator):
        latent = Input(shape=(self.z_dim,))
        image = generator(latent)
        prediction = discriminator(image)

        return Model(input=latent, output=prediction)

    def _load_dataset(self):
        # NOTE GAN stability hack: Normalize images between -1 and +1
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = np.concatenate(X_train, X_test, axis=0)
        data = (X_train.astype('float32') - 255.0/2) / (255.0/2)

        return data

    def train(self, batch_size=128, epochs, before_epoch=None, after_epoch):
        # Initialize optimizers and models
        adam = Adam(lr=0.0003)
        sgd = SGD(lr=0.003)
        self.generator = generator = _generator()
        discriminator = _discriminator()
        GAN = _GAN(generator, discriminator)

        # NOTE GAN advice: Adam for Generator, vanilla SGD for Discriminator
        generator.compile(loss='binary_crossentropy', optimizer=adam)
        discriminator.compile(loss='binary_crossentropy', optimizer=SGD)
        GAN.compile(loss='binary_crossentropy', optimizer=adam)

        # Track losses
        loss_discriminator = []
        loss_gan = []

        # Train
        dataset = _load_data()
        dataset_size = dataset.shape()[0]
        batches = dataset_size/batch_size)

        for epoch in range(1, epochs + 1):
            if before_epoch is not None:
                before_epoch(epoch)

            for i in range(batches):
                # NOTE GAN advice: Sample latent vectors from a Unit Gaussian instead of Uniform
                latents = np.random.gaussian(0, 1, size=(batch_size, self.z_dim))
                # Sample images from the Generator
                samples = generator.predict(latents)
                # Sample images from the Dataset
                reals = dataset[np.random.randint(dataset_size, size=batch_size)]
                # Create labeled dataset for the Discriminator and GAN
                X = np.random.shuffle(np.concatenate((samples, reals), axis=0))
                Y = np.random.shuffle(np.concatenate(
                        (np.zeros(batch_size).astype(int), np.ones(batch_size).astype(int)), axis=0
                    ))
                Z = np.ones(batch_size).astype(int)
                # Train Discriminator
                discriminator.trainable = True
                loss_d = discriminator.train_on_batch(X, Y)
                # Train Generator
                discriminator.trainable = False
                loss_g = GAN.train_on_batch(latents, Z)

                loss_discriminator.append(loss_d)
                loss_gan.append(loss_g)


            if after_epoch is not None:
                after_epoch(
                    epoch,
                    loss_gan=loss_g,
                    loss_discriminator=loss_d
                    )
