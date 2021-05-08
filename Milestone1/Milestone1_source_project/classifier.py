from keras.layers import Input, Dense, Reshape, Dropout,Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
import keras.backend as K
from DataLoader import DataLoader
from DataHandler import DataHandler
import os

def build_classifier(train_data):
    num_train, height, width, channel = train_data.shape


    model = Sequential()
    model.add(Convolution2D(50, (1 , 5), padding='valid', activation='relu', input_shape=(height, width,1)))
    model.add(Convolution2D(50, (1 , 3), padding='same', activation='relu'))

    model.add(Dense(50, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(40, (1 , 5), padding='valid', activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(40, (1 , 3), padding='valid', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense((400), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(6, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


def main():
    
    
    
    features = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
    act_labels = ["dws","ups","wlk", "jog", "sit", "std"]


    train_loader = DataLoader()
    train_ts, test_ts,num_features, num_act_labels = train_loader.ts(features, act_labels)
    
    
    train_data, act_train_labels, train_mean, train_std = train_loader.time_series_to_section(train_ts.copy(),
                                                                                                   num_act_labels,
                                                                                                   sliding_window_size=200,
                                                                                                   step_size_of_sliding_window=10)
    
    test_data, act_test_labels, train_mean, train_std = train_loader.time_series_to_section(test_ts.copy(),
                                                                                                   num_act_labels,
                                                                                                   sliding_window_size=200,
                                                                                                   step_size_of_sliding_window=10)
    
    handler = DataHandler(train_data, test_data)
    norm_train = handler.normalise("train")
    norm_test = handler.normalise("test")

    print("--- Shape of Training Data:", train_data.shape)
    print("--- Shape of Test Data:", test_data.shape)


    BATCH_SIZE = 64
    EPOCHS = 20
    VERBOSITY = True
    
    classifier = build_classifier(norm_train)
    classifier.fit(norm_train, [act_train_labels],                
                               batch_size = BATCH_SIZE,
                               epochs = EPOCHS,
                               verbose = VERBOSITY)
    
    results = classifier.evaluate(norm_test, act_test_labels,
                                         verbose = VERBOSITY)
    
    
    print("--> Evaluation on Test Dataset:")
    print("**** Accuracy for Activity Recognition Task is: ", results[1])
    
    if not os.path.exists('pretrained_models'):
        os.makedirs('pretrained_models')

    classifier.save("pretrained_models/classifier.hdf5")

if __name__ == "__main__":
    main()