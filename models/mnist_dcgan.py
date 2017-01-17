import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.optimizers import SGD, Adam
from keras.datasets import mnist

class MNIST_DCGAN:
    def __init__(self, z_dim):
        self.input_dim = (None, 28, 28)
        self.z_dim = 100
        self.nf = 64 # filters count
        self.generator = None

    def _generator(self):
        return Sequential([
            Dense(1024, input_shape=(self.z_dim,), input_dim=self.z_dim, init='glorot_normal'),
            BatchNormalization(),
            LeakyReLU(alpha=0.3),
            Dense(self.nf * 7 * 7, init='glorot_normal'),
            BatchNormalization(),
            LeakyReLU(alpha=0.3),
            Reshape((7, 7, self.nf)),
            Deconvolution2D(self.nf, 5, 5, output_shape=(None, 14, 14, self.nf), subsample=(2,2), border_mode='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.3),
            Deconvolution2D(1, 5, 5, output_shape=(None, 28, 28, 1), subsample=(2,2), border_mode='same'),
            Activation('tanh')
        ])

    def _discriminator(self):
        return Sequential([
            Convolution2D(self.nf, 5, 5, input_shape=(28, 28, 1), subsample=(2,2), border_mode='same', init='glorot_normal'),
            LeakyReLU(alpha=0.2),
            Convolution2D(self.nf * 2, 5, 5, subsample=(2,2), border_mode='same', init='glorot_normal'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1, init='glorot_normal'),
            Activation('sigmoid')
        ])

    def _GAN(self, generator, discriminator):
        return Sequential([
            generator,
            discriminator
        ])

    def _load_dataset(self):
        # NOTE GAN stability hack: Normalize images between -1 and +1
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        data = np.concatenate((X_train, X_test), axis=0)
        data = (data.astype('float32') - 255.0/2) / (255.0/2)
        data = data.reshape(data.shape + (1,))

        return data

    def train(self, batch_size, epochs, before_epoch=None, after_epoch=None):
        # Initialize optimizers and models
        adam = Adam(lr=0.0003)
        sgd = SGD(lr=0.003)

        generator = self.generator = self._generator()
        discriminator = self._discriminator()
        GAN = self._GAN(generator, discriminator)

        # NOTE GAN advice: Adam for Generator, vanilla SGD for Discriminator
        generator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        discriminator.compile(loss='binary_crossentropy', optimizer=SGD, metrics=['accuracy'])

        discriminator.trainable = False
        GAN.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Track losses
        loss_discriminator = []
        loss_gan = []

        # Train
        dataset = self._load_dataset()
        dataset_size = dataset.shape[0]
        batches = int(dataset_size / batch_size)

        for epoch in range(1, epochs + 1):
            if before_epoch is not None:
                before_epoch(epoch)

            for i in range(batches):
                # NOTE GAN advice: Sample latent vectors from a Unit Gaussian instead of Uniform
                latents = np.random.normal(0, 1, size=(batch_size, self.z_dim))
                # Sample images from the Generator
                samples = generator.predict(latents)
                # Sample images from the Dataset
                reals = dataset[np.random.randint(dataset_size, size=batch_size)]
                # Create labeled dataset for the Discriminator and GAN
                X = np.concatenate((samples, reals), axis=0)
                Y = np.concatenate((np.zeros(batch_size).astype(int), np.ones(batch_size).astype(int)), axis=0)
                rand = np.arange(len(X))
                np.random.shuffle(rand)
                X = X[rand]
                Y = Y[rand]
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
                after_epoch(epoch, loss_gan=loss_g, loss_discriminator=loss_d)
