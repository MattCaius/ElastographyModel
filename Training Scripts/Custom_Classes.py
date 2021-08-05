import keras.engine
import tensorflow as tf
import numpy as np
from Utilities import GenFramePair
from tensorflow.keras import activations
from tensorflow.keras import backend as K

# Randomly translates if training, else just crop 10% off each side

class RandomTranslator(tf.keras.layers.Layer):
    def __init__(self, batch_size, baseCut = 0.10):
        super(RandomTranslator, self).__init__()
        self.baseCut = baseCut
        self.lowerI = int(self.baseCut * 256)
        self.upperI = int((1 - self.baseCut) * 256)
        self.lowerJ = int(self.baseCut * 2062)
        self.upperJ = int((1 - self.baseCut) * 2062)
        self.batch_size = batch_size

    def call(self, pairs, training = False):

        output = list()

        for i in range(self.batch_size):

            pair = pairs[i, :, :, :]

            if not training:
                output.append(tf.expand_dims(tf.slice(pair, [self.lowerJ,self.lowerI, 0],
                                                      [self.upperJ - self.lowerJ, self.upperI - self.lowerI, 2]),axis=0))
            else:
                origin = [
                    np.random.randint(0, self.lowerJ),
                    np.random.randint(0, self.lowerI),
                    0
                ]

                output.append(tf.expand_dims(tf.slice(pair, origin, (self.upperJ - self.lowerJ, self.upperI - self.lowerI, 2)), axis = 0))

        return tf.concat(output, 0)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'baseCut' : self.baseCut,
            'lowerI' : self.lowerI,
            'lowerJ' : self.lowerJ
        })
        return config

# Dataloader to load the frame pairs into the model
# rawData is of shape [patient_index, i, j, frame_no]
# DOES NOT SHUFFLE - SHUFFLE DIRECTORY WHEN GENERATING IT

class RF_PairLoader(tf.keras.utils.Sequence):

    def __init__(self, directory, rawData,
                 scan_col, i_col, j_col, labels,
                 batch_size, is3D = True):

        self.dir = directory
        self.rawData = rawData
        self.scan_col = scan_col
        self.i_col = i_col
        self.j_col = j_col
        self.labels = labels
        self.batch_size = batch_size
        self.is3D = is3D

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

        if self.is3D:
            x_batch = tf.expand_dims(tf.concat(pairs, 0), -1)
        else:
            x_batch = tf.concat(pairs, 0)
        y_batch = self.dir[self.labels][idx]

        return np.asarray(x_batch), np.asarray(y_batch)


    def __len__(self):
        return self.n // self.batch_size

# This is the correlation layer for the flownet - needs to be custom as keras does not support it

class Normalized_Correlation_Layer(tf.keras.layers.Layer):

    def __init__(self, patch_size = (5,5),
        dim_ordering = 'tf',
        border_mode = 'same',
        stride = (1,1),
        activation = None,
        **kwargs):

        if border_mode != 'same':
            raise ValueError('Invalid border mode for Correlation Layer '
                             '(only "same" is supported as of now):', border_mode)
        self.kernel_size = patch_size
        self.subsample = stride
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.activation = activations.get(activation)
        super(Normalized_Correlation_Layer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2],
                self.kernel_size[0] * input_shape[0][2] * input_shape[0][-1])

    def get_config(self):
        config = {'patch_size': self.kernel_size,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'stride': self.subsample,
                  'dim_ordering': self.dim_ordering}
        base_config = super(Normalized_Correlation_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def Call(self, x, mask=None):
        input_1, input_2 = x
        stride_row, stride_col = self.subsample
        inp_shape = input_1._keras_shape

        output_shape = self.compute_output_shape([inp_shape, inp_shape])

        padding_row = (int(self.kernel_size[0] / 2), int(self.kernel_size[0] / 2))
        padding_col = (int(self.kernel_size[1] / 2), int(self.kernel_size[1] / 2))
        input_1 = K.spatial_2d_padding(input_1, padding=(padding_row, padding_col))
        input_2 = K.spatial_2d_padding(input_2, padding=((padding_row[0] * 2, padding_row[1] * 2), padding_col))

        output_row = output_shape[1]
        output_col = output_shape[2]

        output = []

        for k in range(inp_shape[-1]):
            xc_1 = []
            xc_2 = []
            for i in range(padding_row[0]):
                for j in range(output_col):
                    xc_2.append(K.reshape(input_2[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1], k],
                                          (-1, 1, self.kernel_size[0] * self.kernel_size[1])))

        for i in range(output_row):
            slice_row = slice(i, i + self.kernel_size[0])
            slice_row2 = slice(i + padding_row[0], i + self.kernel_size[0] + padding_row[0])
            for j in range(output_col):
                slice_col = slice(j, j + self.kernel_size[1])
                xc_2.append(K.reshape(input_2[:, slice_row2, slice_col, k],
                                      (-1, 1, self.kernel_size[0] * self.kernel_size[1])))
                xc_1.append(K.reshape(input_1[:, slice_row, slice_col, k],
                                      (-1, 1, self.kernel_size[0] * self.kernel_size[1])))

        for i in range(output_row, output_row + padding_row[1]):
            for j in range(output_col):
                xc_2.append(K.reshape(input_2[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1], k],
                                      (-1, 1, self.kernel_size[0] * self.kernel_size[1])))

        xc_1_aggregate = K.concatenate(xc_1, axis=1)

        xc_1_mean = K.mean(xc_1_aggregate, axis=-1, keepdims=True)
        xc_1_std = K.std(xc_1_aggregate, axis=-1, keepdims=True)
        xc_1_aggregate = (xc_1_aggregate - xc_1_mean) / xc_1_std

        xc_2_aggregate = K.concatenate(xc_2, axis=1)
        xc_2_mean = K.mean(xc_2_aggregate, axis=-1, keepdims=True)
        xc_2_std = K.std(xc_2_aggregate, axis=-1, keepdims=True)
        xc_2_aggregate = (xc_2_aggregate - xc_2_mean) / xc_2_std

        xc_1_aggregate = K.permute_dimensions(xc_1_aggregate, (0, 2, 1))
        block = []
        len_xc_1 = len(xc_1)
        for i in range(len_xc_1):
            # This for loop is to compute the product of a given patch of feature map 1 and the feature maps on which it is supposed to
            sl1 = slice(int(i / inp_shape[2]) * inp_shape[2],
                        int(i / inp_shape[2]) * inp_shape[2] + inp_shape[2] * self.kernel_size[0])
            # This calculates which are the patches of feature map 2 to be considered for a given patch of first feature map.

            block.append(K.reshape(K.batch_dot(xc_2_aggregate[:, sl1, :],
                                               xc_1_aggregate[:, :, i]),
                                   (-1, 1, 1, inp_shape[2] * self.kernel_size[0])))

        block = K.concatenate(block, axis=1)
        block = K.reshape(block, (-1, output_row, output_col, inp_shape[2] * self.kernel_size[0]))
        output.append(block)

        output = self.activation(output)
        return output