import logging
import numpy as np

from keras import Input, Model
from keras.layers import LSTM, Dense, Reshape, Dropout, concatenate
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class RNNTossing:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Learning tossing using RNN...")

        all_data = pad_sequences(np.array(self._c['data_coded']), maxlen=100)
        all_owner = np.array(self._c['target_coded'])

        logger.info('loaded data: %d' % len(all_data))

        num_classes = len(self._c['target_names'])

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        cvscores = []
        for train, test in kfold.split(all_data, all_owner):
            x_train = all_data[train]
            y_train = np_utils.to_categorical(all_owner[train], num_classes)
            x_test = all_data[test]
            y_test = np_utils.to_categorical(all_owner[test], num_classes)

            sequence = Input(shape=(100,), dtype='float32')
            reshape = Reshape((1, 100))(sequence)
            forwards_1 = LSTM(1024)(reshape)
            after_dp_forward_4 = Dropout(0.20)(forwards_1)
            backwards_1 = LSTM(1024, go_backwards=True)(reshape)
            after_dp_backward_4 = Dropout(0.20)(backwards_1)
            merged = concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
            after_dp = Dropout(0.5)(merged)
            output = Dense(len(self._c['target_names']), activation='softmax')(after_dp)
            model = Model(input=sequence, output=output)
            rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
            model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1)

            scores = model.evaluate(x_test, y_test, verbose=0, batch_size=32)

            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            model.evaluate(x_test, y_test)

        return self._c

    def load(self):
        storage = self._c['storage']

        data_rows = storage.load_bugs_and_history()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._c['data'] = []
        self._c['target'] = []
        self._c['target_names'] = list(set(developer_ids))

        assignee_names = []

        for bug in data_rows:

            vector = bug['summary']
            history = bug['history']

            for v in bug['comments']:
                vector += str(v) + ' '

            path = []
            for assign in history:
                path.append(assign['email'])
                assignee_names.append(assign['email'])

            self._c['data'].append(path)
            self._c['target'].append(bug['assignee_email'])

        assignee_names = list(set(assignee_names))

        self._c['target_coded'] = [self._c['target_names'].index(target) for target in
                                   self._c['target']]

        self._c['data_coded'] = [[assignee_names.index(tossee) for tossee in path] for path in
                                 self._c['data']]

        return self._c
