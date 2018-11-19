import logging
from decimal import Decimal

import numpy as np

import re, nltk, string

from gensim.models import Word2Vec
from keras.engine import InputSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, concatenate, BatchNormalization, merge, \
    TimeDistributed, Wrapper, K
from keras.optimizers import RMSprop
from keras.utils import np_utils

from src.base import Trainer
from src.tools.mappers.bug_vector import get_bug_vector_comment, clean_item
from src.tools.progress_bar import print_progressbar
from src.tools.time_estimator import FixTimeEstimator

np.random.seed(1337)

logger = logging.getLogger(__name__)

# 1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

# 2. Classifier hyperparameters
numCV = 10
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 32


class MSRDeepTriage(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container
        self._ft_estimator = None

    def train(self):
        logger.info("Training using original DeepTriage")

        all_data = np.array(self._c['data'])
        all_owner = np.array(self._c['target'])

        wordvec_model = Word2Vec(sentences=self._c['documents'],
                                 min_count=min_word_frequency_word2vec,
                                 size=embed_size_word2vec, window=context_window_word2vec)
        vocabulary = wordvec_model.wv.vocab

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                all_data, all_owner, range(0, len(self._c['data'])),
                test_size=0.2, random_state=42)

        all_bug_ids = self._c['bug_ids']
        bug_ids = []
        train_data = all_data[indices_train]
        test_data = all_data[indices_test]
        train_owner = all_owner[indices_train]
        test_owner = all_owner[indices_test]

        updated_train_data = []
        updated_train_data_length = []
        updated_train_owner = []
        final_test_data = []
        final_test_owner = []
        for j, item in enumerate(train_data):
            current_train_filter = [word for word in item if word in vocabulary]
            if len(current_train_filter) >= min_sentence_length:
                updated_train_data.append(current_train_filter)
                updated_train_owner.append(train_owner[j])

        for j, item in enumerate(test_data):
            current_test_filter = [word for word in item if word in vocabulary]
            if len(current_test_filter) >= min_sentence_length:
                final_test_data.append(current_test_filter)
                final_test_owner.append(test_owner[j])
                bug_ids.append(all_bug_ids[j])

        # Remove data from test set that is not there in train set
        train_owner_unique = set(updated_train_owner)
        test_owner_unique = set(final_test_owner)
        unwanted_owner = list(test_owner_unique - train_owner_unique)
        updated_test_data = []
        updated_test_owner = []
        updated_test_data_length = []
        updated_bug_ids = []

        for j in range(len(final_test_owner)):
            if final_test_owner[j] not in unwanted_owner:
                updated_test_data.append(final_test_data[j])
                updated_test_owner.append(final_test_owner[j])
                updated_bug_ids.append(bug_ids[j])

        unique_train_label = list(set(updated_train_owner))
        classes = np.array(unique_train_label)

        # Create train and test data for deep learning + softmax
        X_train = np.empty(
            shape=[len(updated_train_data), max_sentence_len, embed_size_word2vec],
            dtype='float32')
        Y_train = np.empty(shape=[len(updated_train_owner), 1], dtype='int32')
        # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
        for j, curr_row in enumerate(updated_train_data):
            sequence_cnt = 0
            for item in curr_row:
                if item in vocabulary:
                    X_train[j, sequence_cnt, :] = wordvec_model[item]
                    sequence_cnt = sequence_cnt + 1
                    if sequence_cnt == max_sentence_len - 1:
                        break
            for k in range(sequence_cnt, max_sentence_len):
                X_train[j, k, :] = np.zeros((1, embed_size_word2vec))
            Y_train[j, 0] = unique_train_label.index(updated_train_owner[j])

        X_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec],
                          dtype='float32')
        Y_test = np.empty(shape=[len(updated_test_owner), 1], dtype='int32')

        final_bug_ids = []
        # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
        for j, curr_row in enumerate(updated_test_data):
            sequence_cnt = 0
            for item in curr_row:
                if item in vocabulary:
                    X_test[j, sequence_cnt, :] = wordvec_model[item]
                    sequence_cnt = sequence_cnt + 1
                    if sequence_cnt == max_sentence_len - 1:
                        break
            for k in range(sequence_cnt, max_sentence_len):
                X_test[j, k, :] = np.zeros((1, embed_size_word2vec))
            final_bug_ids.append(updated_bug_ids[j])
            Y_test[j, 0] = unique_train_label.index(updated_test_owner[j])

        y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
        y_test = np_utils.to_categorical(Y_test, len(unique_train_label))

        # Construct the deep learning model
        sequence = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
        forwards_1 = LSTM(1024)(sequence)
        after_dp_forward_4 = Dropout(0.20)(forwards_1)
        backwards_1 = LSTM(1024, go_backwards=True)(sequence)
        after_dp_backward_4 = Dropout(0.20)(backwards_1)
        merged = concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
        after_dp = Dropout(0.5)(merged)
        output = Dense(len(unique_train_label), activation='softmax')(after_dp)
        model = Model(input=sequence, output=output)
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

        model.compile(loss='categorical_crossentropy', optimizer=rms,
                      metrics=['accuracy'])
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=200, verbose=1)

        predict = model.predict(X_test)
        accuracy = []
        sorted_indices = []

        for ll in predict:
            sorted_indices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
        for k in range(1, rankK + 1):
            id = 0
            true_num = 0
            for sorted_ind in sorted_indices:
                if Y_test[id] in sorted_ind[:k]:
                    true_num += 1
                id += 1
            accuracy.append((float(true_num) / len(predict)) * 100)

        pred_classes = [classes[it] for it in sorted_indices]
        print('Test accuracy:', accuracy)

        top_1 = [item[0] for item in sorted_indices]
        prec, recall, fscore, _ = precision_recall_fscore_support(Y_test, top_1,
                                                                  average="weighted")
        print(
            "precision, recall, fscore (macro average): {:.2f}, {:.2f}, {:.2f}".format(prec, recall,
                                                                                       fscore))

        ft_change = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_ft = 0

        for index, bug_id in enumerate(final_bug_ids):
            fix_time = self._ft_estimator.actual_ft(bug_id)
            total_ft += fix_time

            lowest_ft = None

            for _index in range(len(ft_change)):
                bug_index = self._c['bug_ids'].index(bug_id)
                estimation = self._ft_estimator.predict(pred_classes[index].tolist()[_index],
                                                        " ".join(self._c['data'][bug_index]))
                change = fix_time

                if _index == 0:
                    lowest_ft = change

                if estimation and pred_classes[index].tolist()[_index] != self._c['target'][index]:
                    change = estimation

                if change < lowest_ft:
                    lowest_ft = change

                ft_change[_index] += lowest_ft

        fix_time_perc = [str(round(Decimal(1 - ft / total_ft), 2)) for ft in ft_change]

        print('fix time reduction [%]: ' + ', '.join(fix_time_perc))

    def load(self):
        storage = self._c['storage']

        data_rows = storage.load_bugs_and_history()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._c['data'] = []
        self._c['documents'] = []
        self._c['target'] = []
        self._c['bug_ids'] = []
        self._c['target_names'] = list(set(developer_ids))

        for item in data_rows:
            current_data = clean_item(item['title']) + clean_item(item['summary'])

            if 'FIXED' in item['status']:
                self._c['data'].append(current_data)
                self._c['bug_ids'].append(item['id'])
                self._c['target'].append(item['assignee_email'])

            self._c['documents'].append(current_data)

        tfidf = TfidfVectorizer(lowercase=True,
                                stop_words='english',
                                analyzer='word')

        tfidf.fit([" ".join(it) for it in self._c['data']])

        self._ft_estimator = FixTimeEstimator(data_rows, tfidf.transform)
