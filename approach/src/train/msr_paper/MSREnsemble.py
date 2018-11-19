import time
import logging
import pickle
import os
import numpy as np
from decimal import Decimal

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.base import Trainer
from src.tools.decorators.cached import cached
from src.tools.progress_bar import print_progressbar
from src.tools.time_estimator import FixTimeEstimator
from src.tools.tossing.create_dir_graph import create_directed_graph
from src.tools.tossing.developer_activity import get_events_per_developer
from src.tools.tossing.predict_tossee import predict_tossee
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


top_k = 10
top_k_padding = 5

flatten = lambda l: [item for sublist in l for item in sublist]


def concat_prediction(predictions):
    return [
        " ".join(pred_list) for pred_list in predictions
    ]


n_splits = 10


# Multifeature Bug Tossing according to Bhattacharya et al. (2012)
class MSREnsemble(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container
        self._clf = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                        min_df=15,
                                                        stop_words='english',
                                                        analyzer='word')),
                              ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01)))])

        self._ensemble = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                             stop_words='english',
                                                             analyzer='word')),
                                   ('clf',
                                    OneVsRestClassifier(MultinomialNB()))])

        self._graph = None
        self._ft_estimator = None

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
        self._graph, node_sizes, glabels = create_directed_graph(goal_oriented_adjacencym, a_length,
                                                                 target_dict, dev_activity)

        for bug in self._c['data']:
            self.update_graph(bug)

        logger.info(
            'Building graph took {:.1f} seconds'.format(time.time() - start_time))

        self._clf, ensemble_x, ensemble_y = self.kfold_train(x_train, y_train)

        self.evaluate(self.predict, indices_test, 'regular')

        logger.info('Training Ensemble learner')

        self._ensemble = self.train_ensemble(ensemble_x, ensemble_y)

        self.evaluate(self.ensemble_predict, indices_test, 'ensemble')

        return self._c

    @cached('ensemble.pkl')
    def train_ensemble(self, ensemble_x, ensemble_y):
        ensemble_x = concat_prediction(ensemble_x)
        self._ensemble.fit(ensemble_x, ensemble_y)
        return self._ensemble

    def ensemble_predict(self, bug):
        prediction = self.predict(bug)

        x_transformed = concat_prediction([prediction])
        probs = self._ensemble.predict_proba(x_transformed)

        k = top_k_padding * top_k  # padding for tossees

        top_k_probas = sorted(zip(self._ensemble.classes_, probs[0]), key=lambda x: x[1],
                              reverse=True)[:k]
        top_k_pred = [proba[0] for proba in top_k_probas]

        results = []

        for pred_index in top_k_pred:
            results.append(prediction[pred_index])

        return results

    @cached('clf.pkl')
    def kfold_train(self, x_train, y_train):
        ensemble_x = []
        ensemble_y = []

        all_data, all_owner = self._c['data'], self._c['target']
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

    def predict(self, bug):
        dev_activity = self._c['events_per_developer']

        probs = self._clf.predict_proba([bug['vector']])

        k = top_k_padding * top_k  # padding for tossees

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

    def evaluate(self, prediction_func, test_indices, prefix=''):
        top_5_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
        ft_change = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_ft = 0

        predictions_x = []
        predictions_y = []

        total = len(test_indices)
        print('evaluating ' + prefix + ' ...')
        for count, index in enumerate(test_indices):
            print_progressbar(count, total)

            test_x = self._c['data'][index]
            test_y = self._c['target'][index]

            prediction = prediction_func(test_x)

            if test_y in prediction:
                hit = prediction.index(test_y)
                predictions_x.append(prediction)
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
                    change += fix_time - estimation

                if change < lowest_ft:
                    lowest_ft = change

                ft_change[_index] += lowest_ft

        top_5_acc_perc = [str(round(Decimal(acc / total), 2)) for acc in top_5_acc]
        fix_time_perc = [str(round(Decimal(1 - ft / total_ft), 2)) for ft in ft_change]

        print('\n' + prefix + ' accuracy: ' + ', '.join(top_5_acc_perc))
        print(prefix + ' fix time reduction [%]: ' + ', '.join(fix_time_perc))

        return predictions_x, predictions_y

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
        data_rows = [bug for bug in data_rows if 'FIXED' in bug['status']]

        for bug in data_rows:
            developer = bug['assignee_email']

            vector = bug['summary']
            vector += bug['title']

            for v in bug['comments']:
                vector += str(v) + ' '

            bug_id = bug['id']
            history = bug['history']

            path = TossingPath(bug_id, developer)
            for assign in history:
                ln = len(target_dict)

                if assign['email'] != 'NA':
                    if assign['email'] not in target_dict:
                        target_dict[assign['email']] = ln

                    path.add_path_item((assign['email'], assign['date']))

            path.get_actual_path(target_dict)

            self._c['data'].append({
                'modified_at': bug['modified_at'],
                'bug_id': bug_id,
                'vector': vector,
                'component': bug['component'],
                'path': path
            })

            self._c['target'].append(developer)

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
