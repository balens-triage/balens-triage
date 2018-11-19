import logging
import nltk
import re
import string

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Input, Embedding, Conv1D, Lambda, BatchNormalization, \
    Activation, MaxPooling1D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# https://github.com/yuhsinliu1993/VDCNN/blob/master/run.py

class VDCNN:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Training using CNN with learnt Embeddings")

        all_data = np.array(self._c['data'])
        all_owner = np.array(self._c['target_coded'])

        # integer encode the documents
        t = Tokenizer()
        t.fit_on_texts(all_data)
        vocab_size = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(all_data)

        embedding_dim = 256

        num_filters = [64, 128, 256, 512]

        drop = 0.5
        top_k = 3
        # pad documents to a max length of 4 words
        max_sequence_length = 512
        padded_docs = pad_sequences(encoded_docs, maxlen=max_sequence_length, padding='post')

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cvscores = []

        for train, test in kfold.split(padded_docs, all_owner):
            x_train = padded_docs[train]
            y_train = np_utils.to_categorical(all_owner[train])
            x_test = padded_docs[test]
            y_test = np_utils.to_categorical(all_owner[test])

            inputs = Input(shape=(max_sequence_length,), dtype='int32', name='inputs')

            embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                  input_length=max_sequence_length)(inputs)

            # First conv layer
            conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedding)

            # Each ConvBlock with one MaxPooling Layer
            for i in range(len(num_filters)):
                conv = ConvBlockLayer(conv.get_shape().as_list()[1:], num_filters[i])(conv)
                conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

            # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
            def _top_k(x):
                x = tf.transpose(x, [0, 2, 1])
                k_max = tf.nn.top_k(x, k=top_k)
                return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

            k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

            # 3 fully-connected layer with dropout regularization
            fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
            fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
            fc3 = Dense(len(self._c['target_names']), activation='softmax')(fc2)

            # define optimizer
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
            model = Model(inputs=inputs, outputs=fc3)
            model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=1, verbose=1, callbacks=[
                TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True,
                            write_images=True)])

            scores = model.evaluate(x_test, y_test, verbose=0)

            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            del model

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    def load(self):
        storage = self._c['storage']

        data_rows = storage.load_bugs_and_comments()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._c['data'] = []
        self._c['target'] = []
        self._c['target_names'] = list(set(developer_ids))

        for item in data_rows:
            # 1. Remove \r
            if not item.get('title'):
                print(item)
            current_title = item['title'].replace('\r', ' ')
            current_desc = item['summary'].replace('\r', ' ')
            # 2. Remove URLs
            current_desc = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '', current_desc)
            # 3. Remove Stack Trace
            start_loc = current_desc.find("Stack trace:")
            current_desc = current_desc[:start_loc]
            # 4. Remove hex code
            current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
            current_title = re.sub(r'(\w+)0x\w+', '', current_title)
            # 5. Change to lower case
            current_desc = current_desc.lower()
            current_title = current_title.lower()
            # 6. Tokenize
            current_desc_tokens = nltk.word_tokenize(current_desc)
            current_title_tokens = nltk.word_tokenize(current_title)
            # 7. Strip trailing punctuation marks
            current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
            current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
            # 8. Join the lists
            current_data = current_title_filter + current_desc_filter
            current_data = [x for x in current_data if x]

            self._c['data'].append(current_data)
            self._c['target'].append(item['assignee_email'])

        self._c['target_coded'] = [self._c['target_names'].index(target) for target in
                                   self._c['target']]


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
