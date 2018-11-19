import networkx as nx
import time
import logging
import pickle
import os
import numpy as np
from decimal import Decimal
from datetime import datetime

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.base import Trainer
from src.tools.decorators.cached import cached
from src.tools.progress_bar import print_progressbar
from src.tools.time_estimator import FixTimeEstimator
from src.tools.tossing.create_dir_graph import create_directed_graph
from src.tools.tossing.developer_activity import get_events_per_developer
from src.tools.tossing.predict_tossee import bhattacharya_predict
from src.tools.tossing.tossing_path import TossingPath

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def list_from_indices(l, indices):
    return [l[idx] for idx in indices]


# Multifeature Bug Tossing according to Bhattacharya et al. (2012)
class MSRMFTossing(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container
        self._clf = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                        stop_words='english',
                                                        analyzer='word')),
                              ('clf',
                               OneVsRestClassifier(MultinomialNB(alpha=0.01)))])
        self._graph = None,
        self._ft_estimator = None

    def train(self):
        logger.info("Running Multifeature Tossing...")
        start_time = time.time()

        num_folds = 11

        target_dict = self._c['target_dict']
        dev_activity = self._c['events_per_developer']

        x_train, x_vds, y_train, y_vds, indices_train, indices_test = \
            train_test_split(
                self._c['data'], self._c['target'], range(0, len(self._c['data'])),
                test_size=(1 / num_folds), random_state=42)

        logger.info("training data set size: " + str(len(x_train)))
        logger.info("testing  data set size: " + str(len(x_vds)))

        a_length = len(target_dict)
        goal_oriented_adjacencym = np.zeros((a_length, a_length))

        for bug in x_train:
            tss = bug['path']

            if tss is None:
                continue

            t1 = tss.get_goal_oriented_path(target_dict)

            # add goal oriented tossings
            # i.e. tossing A - B - C
            # pair1: A-C ;  pair2 B-C
            for x1 in range(0, len(t1) - 1):
                goal_oriented_adjacencym[t1[x1], t1[-1]] += 1

        # create goal_graph
        self._graph, node_sizes, glabels = self.get_graph(goal_oriented_adjacencym, a_length,
                                                          target_dict, dev_activity)

        logger.info(
            'Building graph took {:.1f} seconds'.format(time.time() - start_time))

        completed_x = []
        completed_y = []

        fold_indices = list(chunks(range(0, len(x_train)), (len(x_train) // num_folds - 1)))

        for fold_num in range(0, len(fold_indices)):
            top_5_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
            total = 0.0
            start_time = time.time()

            train_fold_list = list(fold_indices[fold_num])

            test_fold_list = []
            if fold_num < len(fold_indices) - 1:
                test_fold_list = list(fold_indices[fold_num + 1])

            fold_train_x = list_from_indices(x_train, train_fold_list)
            fold_train_y = list_from_indices(y_train, train_fold_list)

            # TODO make incremental per bug
            x_vec = to_vector_list(fold_train_x)
            self._clf.fit(x_vec + completed_x, fold_train_y + completed_y)
            logger.info(
                'Fit: {:.1f} seconds '.format(time.time() - start_time) + str(
                    len(x_vec + completed_x)) + ' items')

            if fold_num == 0:
                for bug in fold_train_x:
                    self.update_graph(bug)

            pred_time = time.time()
            for index in test_fold_list:
                fold_test_x = x_train[index]
                fold_test_y = y_train[index]

                prediction = self.predict(fold_test_x)

                if fold_test_y in prediction:
                    hit = prediction.index(fold_test_y)
                    for i in range(hit, len(top_5_acc)):
                        top_5_acc[i] += 1.0

                total += 1.0

                # update
                self.update_graph(fold_test_x)

            logger.info(
                'Predict: {:.1f} seconds'.format(time.time() - pred_time))
            completed_x = completed_x + x_vec
            completed_y = completed_y + fold_train_y

            if fold_num < len(fold_indices) - 1:
                top_5_acc_perc = [str(round(Decimal(acc / total), 2)) for acc in top_5_acc]

                logger.info(
                    'accuracy after fold ' + str(fold_num + 1) + ' ' + ', '.join(top_5_acc_perc)
                    + ' in {:.1f} seconds '.format(time.time() - start_time))

        self.evaluate(self.predict, indices_test, 'final')

        return self._c

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
                predictions_x.append(prediction + [test_x['vector']])
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
                    change += estimation

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

    def predict(self, bug):
        dev_activity = self._c['events_per_developer']

        probs = self._clf.predict_proba([bug['vector']])
        top_5_probas = sorted(zip(self._clf.classes_, probs[0]), key=lambda x: x[1], reverse=True)[
                       :5]
        top_5 = [proba[0] for proba in top_5_probas]

        tossee_1 = bhattacharya_predict(self._graph, bug, top_5[0],
                                        developer_activity=dev_activity)
        tossee_2 = bhattacharya_predict(self._graph, bug, top_5[1],
                                        developer_activity=dev_activity)

        return [top_5[0], tossee_1[0]['email'], top_5[1], tossee_2[0]['email'],
                top_5[2], top_5[3]]

    @cached('graph.pkl')
    def get_graph(self, matrix, a_length, target_dict, dev_activity):
        return create_directed_graph(matrix, a_length,
                                     target_dict, dev_activity)

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
        self._c['target_names'] = list(set(developer_ids))
        target_dict = {}
        self._c['target_dict'] = {}

        # check bug 344789 in mozilla
        # fixers do not always show up in the history, because they may have changed
        # their email after fixing the bug
        # Therefore clean up email changes:
        def clean_bug(bug):
            _developer = bug['assignee_email']
            _history = bug['history']

            hist_email = [_assign['email'] for _assign in _history]
            if _developer not in hist_email:
                bug['history'][-1]['email'] = _developer

            return bug

        data_rows = [clean_bug(bug) for bug in data_rows]

        for bug in data_rows:
            if 'FIXED' not in bug['status']:
                continue

            developer = bug['assignee_email']

            vector = bug['summary']
            bug_id = bug['id']
            history = bug['history']

            for v in bug['comments']:
                vector += str(v) + ' '

            self._c['bug_ids'].append(bug_id)
            self._c['target'].append(developer)

            path = TossingPath(bug_id, developer)
            for assign in history:
                ln = len(target_dict)

                if assign['email'] != 'NA':
                    if assign['email'] not in target_dict:
                        target_dict[assign['email']] = ln

                    path.add_path_item((assign['email'], assign['date']))

            self._c['data'].append({
                'modified_at': bug['modified_at'],
                'vector': vector,
                'component': bug['component'],
                'path': path,
                'bug_id': bug_id
            })

        data_rows = [row for row in data_rows if 'FIXED' in row['status']]

        self._c['data'], self._c['target'] = np.array(self._c['data']), np.array(self._c['target'])

        self._c['target_dict'] = target_dict

        self._c['events_per_developer'] = self.get_events(data_rows)

        tfidf = TfidfVectorizer(lowercase=True,
                                stop_words='english',
                                analyzer='word')

        tfidf.fit([it['vector'] for it in self._c['data']])

        self._ft_estimator = FixTimeEstimator(data_rows, tfidf.transform)

        logger.info(
            'Loading data took {:.1f} seconds'.format(time.time() - start_time))

        return self._c

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
