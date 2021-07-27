import tensorflow as tf
import numpy as np
from Utilities import GenFramePair

# Randomly translates if training, else just crop 10% off each side

class RandomTranslator(tf.keras.layers.Layer):
    def __init__(self, baseCut = 0.10):
        super(RandomTranslator, self).__init__()
        self.baseCut = baseCut
        self.lowerI = int(self.baseCut * 256)
        upperI = int((1 - self.baseCut) * 256)
        self.lowerJ = int(self.baseCut * 2064)
        upperJ = int((1 - self.baseCut) * 2064)
        self.cutSize = [upperJ - self.lowerJ, upperI - self.lowerI, 2, 1]

    def Call(self, pairs, training = False):

        output = list()

        for i in range(pairs.shape[0]):

            pair = pairs[i, :, :, :, :]

            if not training:
                output.append(tf.expand_dims(tf.slice(pair, [self.lowerJ,self.lowerI, 0, 0], self.cutSize), axis=0))
            else:
                origin = [
                    np.random.randint(0, self.lowerJ),
                    np.random.randint(0, self.lowerI),
                    0,
                    0
                ]

                print(origin)

                output.append(tf.expand_dims(tf.slice(pair, origin, self.cutSize), axis = 0))

        return tf.concat(output, 0)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'baseCut' : self.baseCut,
            'lowerI' : self.lowerI,
            'lowerJ' : self.lowerJ,
            'cutSize' : self.cutSize
        })
        return config

# Dataloader to load the frame pairs into the model
# rawData is of shape [patient_index, i, j, frame_no]
# DOES NOT SHUFFLE - SHUFFLE DIRECTORY WHEN GENERATING IT

class RF_PairLoader(tf.keras.utils.Sequence):

    def __init__(self, directory, rawData,
                 scan_col, i_col, j_col, labels,
                 batch_size):

        self.dir = directory
        self.rawData = rawData
        self.scan_col = scan_col
        self.i_col = i_col
        self.j_col = j_col
        self.labels = labels
        self.batch_size = batch_size

        self.n = directory.shape[0]

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):

        pairs = list()

        idx = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        for index in idx:
            X = GenFramePair(self.dir[self.i_col][index],
                             self.dir[self.j_col][index],
                             self.rawData, self.dir[self.scan_col][index]
                             )
            pairs.append(X)

        x_batch = tf.expand_dims(tf.concat(pairs, 0), -1)
        y_batch = self.dir[self.labels][idx]

        return np.asarray(x_batch), np.asarray(y_batch)


    def __len__(self):
        return self.n // self.batch_size


