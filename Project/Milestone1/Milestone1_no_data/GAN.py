from keras.layers import Input, Dense, Reshape, Dropout, Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
import keras.backend as K
import numpy as np


class GAN():
    def __init__(self, datashape):
        self.gan = None

        ### Generator
        self.generator = None
        self.generator_optimizer = Adam(0.0002, 0.9)

        ### Discriminator
        self.discriminator = None
        self.discriminator_optimizer = SGD(lr=0.012)

        self.num_samples = datashape[0]
        self.features = datashape[1]
        self.seq_length = datashape[2]

        self.input_shape = (self.features, self.seq_length, 1)
        self.latent_dim = 100

        ### Build network
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

    def build_gan(self):
        self.discriminator.trainable = False
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        validity = self.discriminator(generated_seq)

        model = Model(z, validity)
        model.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Convolution2D(50, (1, 10), padding='valid', activation='relu', input_shape=self.input_shape))
        model.add(Convolution2D(50, (1, 5), padding='same', activation='relu'))

        model.add(Dense(int(50), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(40, (1, 3), padding='valid', activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(20, (1, 3), padding='valid', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense((400), activation='relu'))
        model.add(Dropout(0.4))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy'])

        return model

    def build_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim, ))
        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(128 * 3 * 20))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((3, 20, 128)))

        model.add(UpSampling2D(size=(1, 5)))
        model.add(Convolution2D(128, (1, 3), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D(size=(1, 2)))
        model.add(Convolution2D(1, (1, 5), strides=(1, 1), padding='same'))
        model.add(Activation('tanh'))
        return model

    def generate(self):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        return predictions
