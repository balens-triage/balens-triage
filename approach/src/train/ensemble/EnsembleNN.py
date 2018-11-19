import time
import logging
import os
import numpy as np
from decimal import Decimal

from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Input, Model
from keras.layers import Dropout, Dense, \
    concatenate, LSTM
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.base import Trainer
from src.tools.decorators.cached import cached
from src.tools.decorators.cachedkmodel import cached_kmodel
from src.tools.mappers.bug_vector import get_bug_vector_comment
from src.tools.progress_bar import print_progressbar
from src.tools.time_estimator import FixTimeEstimator
from src.tools.tossing.create_dir_graph import create_directed_graph
from src.tools.tossing.developer_activity import get_events_per_developer
from src.tools.tossing.predict_tossee import predict_tossee
from src.tools.tossing.tossing_path import TossingPath
from src.train.ensemble.layer import get_clf_layer, get_clf_layer_max_length
from src.train.loader.EmbeddingLoader import load_word2vec

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


flatten = lambda l: [item for sublist in l for item in sublist]

n_splits = 10

batch_size = 32


# Multifeature Bug Tossing according to Bhattacharya et al. (2012)
class EnsembleNN(Trainer):
    top_k = 10
    top_k_padding = 10
    _restarting = False

    def __init__(self, namespace, container, *args):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container
        self._clf = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                        min_df=15,
                                                        stop_words='english',
                                                        analyzer='word')),
                              ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01)))])

        self._ensemble = None
        self._ensemble_labels = None
        self._graph = None
        self._ft_estimator = None
        self._word2vec = None
        self._emb_matrix = None

        self._layer = 'lstm'
        if len(args) > 0:
            self._layer = args[0]

        self._max_sequence_length = get_clf_layer_max_length(self._layer)

        logger.info('Text classification layer: %s' % self._layer)

    def train(self):
        logger.info("Running Ensemble on MFTossing...")
        start_time = time.time()

        target_dict = self._c['target_dict']
        dev_activity = self._c['events_per_developer']

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                self._c['data'], self._c['target'], range(0, len(self._c['data'])),
                test_size=0.2, random_state=42)

        logger.info("training data set size: " + str(len(x_train)))
        logger.info("testing  data set size: " + str(len(x_test)))

        a_length = len(target_dict)
        goal_oriented_adjacencym = np.zeros((a_length, a_length))

        for bug in x_train:
            tss = bug['path']

            if tss is None:
                continue

            t1 = tss.get_goal_oriented_path(target_dict)

            for x1 in range(0, len(t1) - 1):
                goal_oriented_adjacencym[t1[x1], t1[-1]] += 1

        # create goal_graph
        self._graph, node_sizes, glabels = self.get_graph(goal_oriented_adjacencym, a_length,
                                                          target_dict, dev_activity)

        for bug in self._c['data']:
            self.update_graph(bug)

        logger.info(
            'Building graph took {:.1f} seconds'.format(time.time() - start_time))

        self._clf, ensemble_x, ensemble_y = self.kfold_train(x_train, y_train)

        self._word2vec, self._emb_matrix = self.prepare_ensemble()

        # self.evaluate(self.predict, indices_test, 'regular')

        logger.info('Training Ensemble learner')
        self._ensemble_labels = np.array(list(set(ensemble_y)))
        self._ensemble = self.train_ensemble(ensemble_x, ensemble_y)

        if not self._restarting:
            self.evaluate(self.ensemble_predict, indices_test, 'ensemble')

        return self._c

    @cached('graph.pkl')
    def get_graph(self, matrix, a_length, target_dict, dev_activity):
        return create_directed_graph(matrix, a_length,
                                     target_dict, dev_activity)

    @cached('clf.pkl')
    def kfold_train(self, x_train, y_train):
        ensemble_x = []
        ensemble_y = []

        all_data, all_owner = np.array(self._c['data']), np.array(self._c['target'])
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        count = 1

        for train, test in kfold.split(x_train, y_train):
            start_time = time.time()

            train_data = all_data[train]
            train_owner = all_owner[train]

            self._clf.fit(to_vector_list(train_data), train_owner)

            pred_x, pred_y = self.evaluate(self.predict, test, 'fold ' + str(count))
            ensemble_x += pred_x
            ensemble_y += pred_y

            logger.info('fold took {:.1f} seconds '.format(time.time() - start_time))
            count += 1

        # final round to use all the data
        self._clf.fit(to_vector_list(x_train), y_train)

        return self._clf, ensemble_x, ensemble_y

    def evaluate(self, prediction_func, test_indices, prefix=''):

        top_5_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
        ft_change = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_ft = 0

        predictions_x = []
        predictions_y = []
        y_pred = []

        total = len(test_indices)
        print('evaluating ' + prefix + ' ...')
        for count, index in enumerate(test_indices):
            print_progressbar(count, total)

            test_x = self._c['data'][index]
            test_y = self._c['target'][index]

            prediction = prediction_func(test_x)

            y_pred.append(prediction[0])

            if test_y in prediction:
                hit = prediction.index(test_y)
                predictions_x.append((prediction, test_x['vector']))
                predictions_y.append(hit)
                for i in range(hit, len(top_5_acc)):
                    top_5_acc[i] += 1.0

            fix_time = self._ft_estimator.actual_ft(test_x['bug_id'])
            total_ft += fix_time
            lowest_ft = None

            for _index in range(len(ft_change)):
                estimation = self._ft_estimator.predict(prediction[_index], test_x['vector'])
                change = fix_time

                if _index == 0:
                    lowest_ft = change

                if estimation and prediction[_index] != test_y:
                    change = estimation

                if change < lowest_ft:
                    lowest_ft = change

                ft_change[_index] += lowest_ft

        top_5_acc_perc = [str(round(Decimal(acc / total), 2)) for acc in top_5_acc]
        fix_time_perc = [str(round(Decimal(1 - ft / total_ft), 2)) for ft in ft_change]

        y_true = self._c['target'][test_indices]

        prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                  average="weighted")
        print(
            "precision, recall, fscore (macro average): {:.2f}, {:.2f}, {:.2f}".format(prec, recall,
                                                                                       fscore))

        print('\n' + prefix + ' accuracy: ' + ', '.join(top_5_acc_perc))
        print(prefix + ' fix time reduction [%]: ' + ', '.join(fix_time_perc))

        return predictions_x, predictions_y

    @cached('ensemble_vars.pkl')
    def prepare_ensemble(self):
        data = []

        for index, bug in enumerate(self._c['data']):
            data.append(self._c['target'][index] + ' ' + bug['vector'])

        data += self._c['rem_bugs']

        # Somehow get all emails into word embedding
        for key, value in self._c['target_dict'].items():
            data.append(key)

        words = [it.split(" ") for it in data]
        model, embedding_matrix = load_word2vec(words)

        return model, embedding_matrix

    @cached_kmodel('ensemble.h5')
    def train_ensemble(self, ensemble_x, ensemble_y):
        ensemble_y_cat = np_utils.to_categorical(ensemble_y)

        pretrained_weights = self._emb_matrix
        vocab_size, embedding_size = pretrained_weights.shape

        # One input for correcting the order of predicted emails
        # and one to fit the bug vector directly
        train_x_pred = np.zeros([len(ensemble_x), self._max_sequence_length], dtype=np.int32)
        train_x_bug = np.zeros([len(ensemble_x), self._max_sequence_length], dtype=np.int32)

        for i, row in enumerate(ensemble_x):
            (predictions, bug) = row
            for t, word in enumerate(predictions):
                if t == self._max_sequence_length:
                    break
                train_x_pred[i, t] = self._word2idx(word)

            for t, word in enumerate(bug.split(' ')):
                if t == self._max_sequence_length:
                    break
                train_x_bug[i, t] = self._word2idx(word)

        train_x_pred = train_x_pred.reshape(len(ensemble_x), self._max_sequence_length, 1)
        train_x_bug = train_x_bug.reshape(len(ensemble_x), self._max_sequence_length, 1)

        input1 = Input(shape=(self._max_sequence_length, 1))
        input2 = Input(shape=(self._max_sequence_length, 1))

        lstm1 = LSTM(embedding_size)(input1)
        dropout1 = Dropout(0.40)(lstm1)

        dl_text_classification = get_clf_layer(self._layer, input2,
                                               embedding_size)

        merged = concatenate([dropout1, dl_text_classification])
        final_dropout = Dropout(0.4)(merged)
        output = Dense(units=ensemble_y_cat.shape[1], activation='softmax')(final_dropout)

        self._ensemble = Model(input=(input1, input2), output=output)

        early_stopper = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0,
                                      mode='auto')

        self._ensemble.compile(loss=self._get_loss(), optimizer='adam',
                               metrics=['accuracy'])

        self._ensemble.fit([train_x_pred, train_x_bug], ensemble_y_cat,
                           batch_size=batch_size,
                           epochs=100, verbose=1, callbacks=[early_stopper])

        return self._ensemble

    def _get_loss(self):
        return 'categorical_crossentropy'

    def _bug_to_vector(self, predictions, bug_vector):
        sentence = " ".join(predictions).split(" ")
        pred_ved = np.zeros((1, self._max_sequence_length), dtype=np.int32)
        bug_vec = np.zeros((1, self._max_sequence_length), dtype=np.int32)
        for t, word in enumerate(sentence):
            if t == self._max_sequence_length:
                break
            elif not self.word_in_vocab(word):
                continue
            pred_ved[0, t] = self._word2idx(word)

        pred_ved = pred_ved.reshape(1, self._max_sequence_length, 1)

        for t, word in enumerate(bug_vector.split(" ")):
            if t == self._max_sequence_length:
                break
            elif not self.word_in_vocab(word):
                continue
            bug_vec[0, t] = self._word2idx(word)

        bug_vec = bug_vec.reshape(1, self._max_sequence_length, 1)

        return pred_ved, bug_vec

    def _word2idx(self, word):
        return self._word2vec.wv.vocab[word].index

    def word_in_vocab(self, word):
        return word in self._word2vec.wv.vocab

    def ensemble_predict(self, bug):
        if not self._ensemble or not self._clf:
            raise Exception('Model not yet trained')

        prediction = self.predict(bug)

        pred_vec, bug_vec = self._bug_to_vector(prediction, bug['vector'])

        probs = self._ensemble.predict([pred_vec, bug_vec])

        k = self.top_k * self.top_k_padding

        top_k_probas = sorted(zip(self._ensemble_labels, probs[0]), key=lambda x: x[1],
                              reverse=True)[:k]

        top_k_pred = [proba[0] for proba in top_k_probas]

        results = []

        for pred_index in top_k_pred:
            results.append(prediction[pred_index])

        return results

    def predict(self, bug):
        dev_activity = self._c['events_per_developer']

        probs = self._clf.predict_proba([bug['vector']])

        k = self.top_k_padding * self.top_k  # padding for tossees

        top_k_probas = sorted(zip(self._clf.classes_, probs[0]), key=lambda x: x[1], reverse=True)[
                       :k]
        top_k_pred = [proba[0] for proba in top_k_probas]

        tossees = [
            predict_tossee(self._graph, bug, dev, developer_activity=dev_activity)[0]['email']
            for dev in top_k_pred]

        result = []

        for index in range(0, len(top_k_pred)):
            result.append(top_k_pred[index])
            result.append(tossees[index])

        return result

    def update_graph(self, bug):
        path = bug['path'].get_id_path()
        for index in range(0, len(path)):
            if index == len(path) - 1:
                break

            source_id = path[index]
            target_id = path[index + 1]
            c = self._graph[source_id][target_id]['components']
            self._graph[source_id][target_id]['components'] = c + [bug['component']]

    def load(self):
        start_time = time.time()
        storage = self._c['storage']

        # bugs ordered by modified_at in SQL statement
        data_rows = storage.load_bugs_and_history()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._c['data'] = []
        self._c['target'] = []
        self._c['bug_ids'] = []
        self._c['rem_bugs'] = []
        self._c['target_names'] = list(set(developer_ids))
        target_dict = {}
        self._c['target_dict'] = {}

        # check bug 344789 in mozilla
        # fixers do not always show up in the history, because they may have changed
        # their email after fixing the bug
        # Therefor clean up email changes:
        def clean_bug(bug):
            _developer = bug['assignee_email']
            _history = bug['history']

            hist_email = [_assign['email'] for _assign in _history]
            if _developer not in hist_email:
                bug['history'][-1]['email'] = _developer

            return bug

        data_rows = [clean_bug(bug) for bug in data_rows]

        for bug in data_rows:
            developer = bug['assignee_email']

            vector = " ".join(get_bug_vector_comment(bug))
            bug_id = bug['id']
            history = bug['history']

            if 'FIXED' in bug['status']:
                self._c['bug_ids'].append(bug_id)

                path = TossingPath(bug_id, developer)
                for assign in history:
                    ln = len(target_dict)

                    if assign['email'] != 'NA':
                        if assign['email'] not in target_dict:
                            target_dict[assign['email']] = ln

                        path.add_path_item((assign['email'], assign['date']))

                self._c['target'].append(developer)
                self._c['data'].append({
                    'modified_at': bug['modified_at'],
                    'assignee_email': bug['assignee_email'],
                    'created_at': bug['created_at'],
                    'bug_id': bug_id,
                    'vector': vector,
                    'component': bug['component'],
                    'path': path,
                    'history': bug['history']
                })
            else:
                self._c['rem_bugs'].append(vector)

        data_rows = [row for row in data_rows if 'FIXED' in row['status']]

        self._c['data'], self._c['target'] = np.array(self._c['data']), np.array(self._c['target'])

        self._c['target_dict'] = target_dict

        self._c['events_per_developer'] = self.get_events(data_rows)

        tfidf = TfidfVectorizer(lowercase=True,
                                min_df=15,
                                stop_words='english',
                                analyzer='word')

        tfidf.fit(to_vector_list(self._c['data']))

        self._ft_estimator = FixTimeEstimator(data_rows, tfidf.transform)

        logger.info(
            'Loading data took {:.1f} seconds'.format(time.time() - start_time))

        return self._c

    def restart(self):
        self.load()
        self._restarting = True
        self.train()

    @cached('events.pkl')
    def get_events(self, data_rows):
        return events_to_last_active_date(
            get_events_per_developer(data_rows))


def to_vector_list(bugs):
    return [bug['vector'] for bug in bugs]


def events_to_last_active_date(events_per_developer):
    def to_last_active_date(events):
        events_sorted = sorted(events, key=lambda event: event[1], reverse=True)

        return events_sorted[0][1]

    return {k: to_last_active_date(v) for k, v in events_per_developer.items()}
