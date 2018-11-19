import time
import logging

import os
import sys
import numpy as np
from datetime import datetime

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, LinearSVR

from src.base import Trainer
from src.tools.decorators.cached import cached
from src.tools.progress_bar import print_progressbar
from src.tools.time_estimator import FixTimeEstimator

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


number_topics = 17


# Multifeature Bug Tossing according to Bhattacharya et al. (2012)
class MSRCosTriage(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._c = container

        self._tfidf = TfidfVectorizer(lowercase=True,
                                      stop_words='english',
                                      analyzer='word')

        self._clf = SVC(kernel="linear", C=1000, probability=True)

        self._lda_model = Pipeline([('lda',
                                     LatentDirichletAllocation(n_topics=number_topics, max_iter=225,
                                                               learning_method='batch', verbose=0,
                                                               evaluate_every=10, n_jobs=6,
                                                               random_state=0))])

        self._graph = None

    def train(self):
        logger.info("Running CosTriage ...")
        start_time = time.time()

        developer_names = self._c['target']
        cbcf_data = self._c['cbcf_data']
        target_names = self._c['target_names']

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                self._c['data'], self._c['target'], range(0, len(self._c['data'])),
                test_size=0.2, random_state=42)

        self._c['data'] = self._tfidf.transform(self._c['data'])
        x_train = self._tfidf.transform(x_train)
        x_test = self._tfidf.transform(x_test)
        self._clf.fit(x_train, y_train)

        predictions = self._clf.predict(x_test)
        print(classification_report(y_test, predictions))

        # create CF space
        # row is a bug id, column is a probability that dev fixes a bug
        LS_space = np.zeros((len(indices_test), len(target_names)))
        index = 0
        for test_index in indices_test:
            Xin = self._c['data'][test_index]
            pred_vec = self._clf.predict_proba(Xin)
            for ci in range(pred_vec.shape[1]):
                LS_space[index][ci] = pred_vec[0][ci]

            index += 1

        logger.info('CF space built.')

        cbcf_svms = self.get_cbcf_svms()

        logger.info('cbcf classifiers trained.')

        lc_space = np.zeros((len(indices_test), len(target_names)))
        index = 0
        ln = len(indices_test)

        for test_index in indices_test:
            if index % 100 == 0:
                print('progress LC %1.2f' % (float(index) * 100 / ln))

            xin = self._c['data'][test_index]
            # iterate each developer's svr and predict cost for a new bug
            for di in range(len(target_names)):
                if target_names[di] in cbcf_svms:
                    regressor = cbcf_svms[target_names[di]]

                    # regressor can be null if there was to little data to train
                    if regressor is not None:
                        bug_cost = regressor.predict(xin)
                        lc_space[index, di] = bug_cost

            index += 1

        logger.info('lc space built.')

        self._lda_model = self.train_lda(self._c['data'])

        logger.info('lda model fit.')

        # dev_profiles = np.zeros((len(target_names), number_topics))
        # dev_profiles_counts = np.zeros((len(target_names), number_topics))
        #
        # logger.info('constructing dev profiles.')
        #
        # total_length = sum(len(cbcf_data[dev]) for dev in developer_names)
        # completed = 0
        # for index, dev in enumerate(developer_names):
        #     dev_set = cbcf_data[dev]
        #     for di in range(len(dev_set)):
        #         print_progressbar(completed, total_length)
        #         bug_vector = dev_set[di][0]
        #         cost = dev_set[di][1]  # time to fix
        #         predicted_type = self._lda_model.transform(bug_vector).argmax()
        #
        #         dev_profiles[target_names.index(dev), predicted_type] += float(cost)
        #         dev_profiles_counts[target_names.index(dev), predicted_type] += 1
        #         completed += 1
        #
        # logger.info('dev profiles constructed.')
        #
        # for t1 in range(len(target_names)):
        #     print_progressbar(t1, len(target_names))
        #     for t2 in range(number_topics):
        #         cost = dev_profiles[t1, t2]
        #         if cost > 0:
        #             cnt = dev_profiles_counts[t1, t2]
        #             dev_profiles[t1, t2] = float(cost) / float(cnt)
        #
        # logger.info('dev profiles cost built.')

        # LC space for costriage
        LC_space2 = np.zeros((len(indices_test), len(X['target_names'])))
        # iterate each bug in test set
        for t1 in range(len(indices_test)):
            # get topic class for this bug
            xhi_in = Xhi[indices_test[t1]]
            predicted_type = lda_model.transform(xhi_in).argmax()
            # dev_profiles2 ,rows are developers, columns are LDA topics, values in matr are costs
            for t2 in range(len(X['target_names'])):
                LC_space2[t1, t2] = dev_profiles2[t2, predicted_type]

        # LC2 space ready, recompute LH
        alphas = [0.2, 0.4, 0.55, 0.7, 0.75, 0.8,  0.9, 0.97, 0.999999]
        for alpha in alphas:
            LH_space = np.zeros((len(indices_test), len(target_names)))

            for t1 in range(LH_space.shape[0]):
                for t2 in range(LH_space.shape[1]):
                    max_ls = np.max(LS_space[t1][np.nonzero(LS_space[t1])])
                    min_lc = np.min(LC_space2[t1][np.nonzero(LC_space2[t1])])
                    if LC_space2[t1, t2] != 0:
                        LH_space[t1, t2] = (alpha * (LS_space[t1, t2] / max_ls)) + (1.0 - alpha)*( (1.0 / LC_space2[t1, t2]) / (1.0 / min_lc) )

            count = 0
            prediction_arr = []

            for t1 in range(len(indices_test)):
                y_true = y_test[t1]
                candidate = -1000000
                ci = 0
                for i2, t2 in enumerate(LH_space[t1]):
                    if  t2 > candidate:
                        candidate = t2
                        ci = i2

                if y_true == ci:
                    count += 1


                prediction_arr.append(ci)

            print('Costriage TOP 1 accuracy: %1.4f' % (float(count) / len(indices_test))

        return self._c

    @cached('lda.pkl')
    def train_lda(self, data):
        self._lda_model.fit(data)
        return self._lda_model

    @cached('cbcf_svms.pkl')
    def get_cbcf_svms(self):
        developer_names = self._c['target']
        cbcf_data = self._c['cbcf_data']

        cbcf_svms = {}
        # train SVM for each developer
        for index, d in enumerate(developer_names):
            print_progressbar(index, len(developer_names))
            dd_set = cbcf_data[d]

            dd_x = []
            dd_y = []

            for completion in dd_set:
                dd_x.append(completion[0].A[0])
                dd_y.append(completion[1])

            if len(dd_y) > 5:
                clf_d = LinearSVR()
                clf_d.fit(dd_x, dd_y)

                cbcf_svms[d] = clf_d

        return cbcf_svms

    def train_pure_cbr(self, x_train, y_train):
        print('ello')

    def load(self):
        start_time = time.time()
        storage = self._c['storage']

        # bugs ordered by modified_at in SQL statement
        data_rows = storage.load_bugs_and_history()

        self._c['target'] = []
        self._c['bug_ids'] = []

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
        vectors = [get_vector(bug) for bug in data_rows]

        self._c['data'] = vectors
        self._tfidf.fit(vectors)
        self._c['cbcf_data'] = self.create_cbcf_data(data_rows)

        for bug in data_rows:
            developer = bug['assignee_email']

            vector = get_vector(bug)
            bug_id = bug['id']

            self._c['bug_ids'].append(bug_id)
            self._c['target'].append(developer)

        self._c['target_names'] = list(set(self._c['target']))

        logger.info(
            'Loading data took {:.1f} seconds'.format(time.time() - start_time))

        return self._c

    @cached('cbcf_data.pkl')
    def create_cbcf_data(self, bugs):
        events_per_developer = {}
        for bug in bugs:
            developer = bug['assignee_email']
            bug_vector = get_vector(bug)
            history = bug['history']
            resolution = bug['resolution']
            idx = -1

            for assign in history:
                idx += 1

                if assign['email'] == developer or idx == (len(history) - 1):
                    start = assign['date']

                    for resolut in resolution:
                        # if resolution is WONTFIX start date and end date are same
                        if resolut['status'].find('FIXED') > -1:
                            # change start time stamp to assignment date for this user
                            end = resolut['time']
                            break
                    else:
                        end = bug['modified_at']

                    event = (self._tfidf.transform([bug_vector]),
                             ((end - start).total_seconds() // (3600 * 24)) + 1)

                    if not events_per_developer.get(assign['email']):
                        events_per_developer[assign['email']] = []

                    events_per_developer[assign['email']].append(event)

        return events_per_developer


def get_vector(bug):
    vector = bug['summary']
    #
    # for v in bug['comments']:
    #     vector += str(v) + ' '

    return vector
