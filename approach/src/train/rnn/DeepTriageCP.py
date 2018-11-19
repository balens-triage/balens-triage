import logging
import numpy as np

import re, nltk, string
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, concatenate, Reshape, Conv2D, MaxPool2D, \
    Flatten, Concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

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


class DeepTriageCP:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Training using updated DeepTriage")

        all_data = np.array(self._c['data'])
        all_owner = np.array(self._c['target'])

        wordvec_model = Word2Vec(sentences=all_data, min_count=min_word_frequency_word2vec,
                                 size=embed_size_word2vec, window=context_window_word2vec)
        vocabulary = wordvec_model.wv.vocab

        kfold = StratifiedKFold(n_splits=11, shuffle=True, random_state=42)
        cvscores = []

        for train, test in kfold.split(all_data, all_owner):

            train_data = all_data[train]
            test_data = all_data[test]
            train_owner = all_owner[train]
            test_owner = all_owner[test]

            updated_train_data = []
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

            # Remove data from test set that is not there in train set
            train_owner_unique = set(updated_train_owner)
            test_owner_unique = set(final_test_owner)
            unwanted_owner = list(test_owner_unique - train_owner_unique)
            updated_test_data = []
            updated_test_owner = []
            updated_test_data_length = []
            for j in range(len(final_test_owner)):
                if final_test_owner[j] not in unwanted_owner:
                    updated_test_data.append(final_test_data[j])
                    updated_test_owner.append(final_test_owner[j])

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

            x_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec],
                              dtype='float32')
            y_test = np.empty(shape=[len(updated_test_owner), 1], dtype='int32')
            # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
            for j, curr_row in enumerate(updated_test_data):
                sequence_cnt = 0
                for item in curr_row:
                    if item in vocabulary:
                        x_test[j, sequence_cnt, :] = wordvec_model[item]
                        sequence_cnt = sequence_cnt + 1
                        if sequence_cnt == max_sentence_len - 1:
                            break
                for k in range(sequence_cnt, max_sentence_len):
                    x_test[j, k, :] = np.zeros((1, embed_size_word2vec))
                y_test[j, 0] = unique_train_label.index(updated_test_owner[j])

            y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
            y_test = np_utils.to_categorical(y_test, len(unique_train_label))

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
            model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
            hist = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             epochs=200, verbose=1)

            scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)

            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            # train_result = hist.history
            del model

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        return self._c

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
