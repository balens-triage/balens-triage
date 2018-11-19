import logging
import numpy as np

import re, nltk, string

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, concatenate, Embedding, Flatten, Concatenate, \
    Conv2D, Reshape, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

from src.base import Trainer

logger = logging.getLogger(__name__)


class CNN(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Training using CNN with learnt Embeddings")

        all_data = np.array(self._c['data'])
        all_owner = np.array(self._c['target_coded'])

        logger.info('loaded data: %d' % len(all_data))

        # integer encode the documents
        t = Tokenizer()
        t.fit_on_texts(all_data)
        vocab_size = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(all_data)

        embedding_dim = 256
        filter_sizes = [3, 4, 5]
        num_filters = 512
        drop = 0.5
        # pad documents to a max length of 4 words
        max_sequence_length = 56
        padded_docs = pad_sequences(encoded_docs, maxlen=max_sequence_length, padding='post')

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cvscores = []

        for train, test in kfold.split(padded_docs, all_owner):
            x_train = padded_docs[train]
            y_train = np_utils.to_categorical(all_owner[train])
            x_test = padded_docs[test]
            y_test = np_utils.to_categorical(all_owner[test])

            inputs = Input(shape=(max_sequence_length,), dtype='int32')
            embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                  input_length=max_sequence_length)(inputs)
            reshape = Reshape((max_sequence_length, embedding_dim, 1))(embedding)

            conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
                            padding='valid', kernel_initializer='normal', activation='relu')(
                reshape)
            conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
                            padding='valid', kernel_initializer='normal', activation='relu')(
                reshape)
            conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
                            padding='valid', kernel_initializer='normal', activation='relu')(
                reshape)

            maxpool_0 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[0] + 1, 1),
                                  strides=(1, 1), padding='valid')(conv_0)
            maxpool_1 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[1] + 1, 1),
                                  strides=(1, 1), padding='valid')(conv_1)
            maxpool_2 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[2] + 1, 1),
                                  strides=(1, 1), padding='valid')(conv_2)

            concatenated_tensor = Concatenate(axis=-1)([maxpool_0, maxpool_1, maxpool_2])
            flatten = Flatten()(concatenated_tensor)
            dropout = Dropout(drop)(flatten)
            output = Dense(units=len(self._c['target_names']), activation='softmax')(dropout)

            # this creates a model that includes
            model = Model(inputs=inputs, outputs=output)

            checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                         monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='auto')
            adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())

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
            if len(current_data) == 0:
                import pdb
                pdb.set_trace()
            self._c['data'].append(current_data)
            self._c['target'].append(item['assignee_email'])

        self._c['target_coded'] = [self._c['target_names'].index(target) for target in
                                   self._c['target']]
