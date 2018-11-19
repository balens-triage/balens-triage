import logging
import numpy as np

import json, re, nltk, string

from keras.callbacks import TensorBoard
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, concatenate
from keras.optimizers import RMSprop
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


class DeepTriage:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Training using original DeepTriage")

        all_data = self._c['data']
        all_owner = self._c['target']

        wordvec_model = Word2Vec(sentences=all_data, min_count=min_word_frequency_word2vec,
                                 size=embed_size_word2vec, window=context_window_word2vec)
        vocabulary = wordvec_model.wv.vocab

        total_length = len(all_data)
        split_length = total_length // (numCV + 1)

        for i in range(1, numCV + 1):
            train_data = all_data[:i * split_length - 1]
            test_data = all_data[i * split_length:(i + 1) * split_length - 1]
            train_owner = all_owner[:i * split_length - 1]
            test_owner = all_owner[i * split_length:(i + 1) * split_length - 1]

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

            X_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec],
                              dtype='float32')
            Y_test = np.empty(shape=[len(updated_test_owner), 1], dtype='int32')
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
            model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
            hist = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             nb_epoch=5, verbose=1,
                             callbacks=[
                                 TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True,
                                             write_images=True)])

            predict = model.predict(X_test)
            accuracy = []
            sorted_indices = []
            pred_classes = []
            for ll in predict:
                sorted_indices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
            for k in range(1, rankK + 1):
                id = 0
                trueNum = 0
                for sortedInd in sorted_indices:
                    pred_classes.append(classes[sortedInd[:k]])
                    if y_test[id] in classes[sortedInd[:k]]:
                        trueNum += 1
                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            print('Test accuracy:', accuracy)

            train_result = hist.history
            del model

        # TODO

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
