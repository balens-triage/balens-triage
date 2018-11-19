import math
from keras.models import Sequential
from keras.layers import LSTM, Dropout, concatenate, Dense, Flatten, Concatenate, MaxPool2D, Conv2D, \
    Reshape, Conv1D, BatchNormalization, Activation, MaxPooling1D, Lambda, GlobalMaxPooling1D


def lstm(input_layer, output_dim):
    lstm2 = LSTM(output_dim)(input_layer)

    return Dropout(0.20)(lstm2)


def deeptriage(input_layer, output_dim):
    forwards_1 = LSTM(1024)(input_layer)
    after_dp_forward_4 = Dropout(0.20)(forwards_1)
    backwards_1 = LSTM(1024, go_backwards=True)(input_layer)
    after_dp_backward_4 = Dropout(0.20)(backwards_1)
    merged = concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
    return Dropout(0.5)(merged)


def cnn(input_layer, output_dim):
    max_sequence_length = input_layer.shape[1]
    num_filters = 512
    filter_sizes = [3, 4, 5]

    dim = math.sqrt(max_sequence_length.value)

    if not dim.is_integer():
        raise ValueError('max_sequence_length ' + str(
            max_sequence_length) + ' needs to have an integer as its square for cnn to work.')
    else:
        dim = int(dim)

    reshape = Reshape((dim, dim, 1))(input_layer)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], dim // 2),
                    padding='valid', kernel_initializer='normal', activation='relu')(
        reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], dim // 2),
                    padding='valid', kernel_initializer='normal', activation='relu')(
        reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], dim // 2),
                    padding='valid', kernel_initializer='normal', activation='relu')(
        reshape)

    maxpool_0 = MaxPool2D(pool_size=(dim - filter_sizes[0] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(dim - filter_sizes[1] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(dim - filter_sizes[2] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=-1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.25)(flatten)
    return Dense(units=output_dim, activation='softmax')(dropout)


def kim(input_layer, output_dim):
    conv_filters = 128  # No. filters to use for each convolution
    # Specify each convolution layer and their kernel siz i.e. n-grams
    conv1_1 = Conv1D(filters=conv_filters, kernel_size=3)(input_layer)
    btch1_1 = BatchNormalization()(conv1_1)
    drp1_1 = Dropout(0.2)(btch1_1)
    actv1_1 = Activation('relu')(drp1_1)
    glmp1_1 = GlobalMaxPooling1D()(actv1_1)

    conv1_2 = Conv1D(filters=conv_filters, kernel_size=4)(input_layer)
    btch1_2 = BatchNormalization()(conv1_2)
    drp1_2 = Dropout(0.2)(btch1_2)
    actv1_2 = Activation('relu')(drp1_2)
    glmp1_2 = GlobalMaxPooling1D()(actv1_2)

    conv1_3 = Conv1D(filters=conv_filters, kernel_size=5)(input_layer)
    btch1_3 = BatchNormalization()(conv1_3)
    drp1_3 = Dropout(0.2)(btch1_3)
    actv1_3 = Activation('relu')(drp1_3)
    glmp1_3 = GlobalMaxPooling1D()(actv1_3)

    conv1_4 = Conv1D(filters=conv_filters, kernel_size=6)(input_layer)
    btch1_4 = BatchNormalization()(conv1_4)
    drp1_4 = Dropout(0.2)(btch1_4)
    actv1_4 = Activation('relu')(drp1_4)
    glmp1_4 = GlobalMaxPooling1D()(actv1_4)

    # Gather all convolution layers
    cnct = concatenate([glmp1_1, glmp1_2, glmp1_3, glmp1_4], axis=1)
    drp1 = Dropout(0.2)(cnct)

    dns1 = Dense(32, activation='relu')(drp1)
    btch1 = BatchNormalization()(dns1)
    drp2 = Dropout(0.2)(btch1)

    return Dense(output_dim, activation='sigmoid')(drp2)


class ConvBlockLayer(object):
    """
    two layer ConvNet. Apply batch_norm and relu after each layer
    """

    def __init__(self, input_shape, num_filters):
        self.model = Sequential()
        # first conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same",
                              input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # second conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def __call__(self, inputs):
        return self.model(inputs)


def vdcnn(input_layer, output_dim):
    num_filters = [64, 128, 256, 512]
    top_k = 3

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(input_layer)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(conv.get_shape().as_list()[1:], num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(x):
        import tensorflow as tf

        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    # 3 fully-connected layer with dropout regularization
    fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
    return Dense(units=output_dim, activation='softmax')(fc2)


layer_switch = {
    'lstm': lstm,
    'deeptriage': deeptriage,
    'cnn': cnn,
    'kim': kim,
    'vdcnn': vdcnn
}

layer_max_lengths = {
    'lstm': 100,
    'deeptriage': 100,
    'cnn': 1024,
    'kim': 1024,
    'vdcnn': 1024
}


# This is the deep learning text classification layer
def get_clf_layer(layer, input_layer, output_dim):
    return layer_switch.get(layer)(input_layer, output_dim)


# This is the deep learning text classification layer
def get_clf_layer_max_length(layer):
    return layer_max_lengths.get(layer)
