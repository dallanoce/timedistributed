# %% [code]
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, list_IDs, input_path, target_path,
                 to_fit=True, batch_size=32, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.input_path = input_path
        self.target_path = target_path
        self.list_IDs = list_IDs
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return [X], y
        else:
            return [X]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = self._load_input(self.input_path, ID)
            X.append(temp)

        X = pad_sequences(X, value=0, padding='post')

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y.append(self._load_target(self.target_path, ID))

        y = pad_sequences(y, value=0, padding='post')

        return y
