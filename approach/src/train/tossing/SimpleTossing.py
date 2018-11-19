import networkx as nx
import time
import numpy as np
# import matplotlib.pyplot as plt
import logging

from pomegranate import DiscreteDistribution, MarkovChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logger = logging.getLogger(__name__)



class SimpleTossing:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("Building tossing graphs...")
        start_time = time.time()

        tossing_path_collection = self._c['tossing']
        logger.info("Found %d paths" % len(tossing_path_collection))

        target_dict = self._c['target_dict']

        logger.info('length of tossing collection is %d' % len(tossing_path_collection))

        train_ratio = 0.8
        train_border = int(len(tossing_path_collection) * train_ratio)
        logger.info('taking %d data to train, %d to test' %
                    (train_border, len(tossing_path_collection) - train_border))

        total = target_dict['__total']
        distribution = {k: v / total for k, v in target_dict.items()}

        distribution.pop('__total', None)

        paths = [t.get_assignee_path() for t in tossing_path_collection]

        # get discrete distribution
        zeroth_dist = DiscreteDistribution(distribution)

        first_chain = MarkovChain.from_samples(paths) # ([zeroth_dist])
        first_chain.fit(paths)

        logger.info(
            'Fitting the paths took {} seconds'.format(time.time() - start_time))

        return self._c

    def load(self):
        storage = self._c['storage']

        data_rows = storage.load_bugs_and_history()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._c['data'] = []
        self._c['target'] = []
        self._c['target_names'] = list(set(developer_ids))
        self._c['target_dict'] = {
            '__total': 0
        }

        def add_dev_name(target_dict, name):
            if not target_dict.get(name):
                target_dict[name] = 1
            else:
                target_dict[name] = target_dict[name] + 1

            target_dict['__total'] += 1

        tossing_path_collection = []

        for bug in data_rows:
            developer = bug['assignee_email']
            vector = bug['summary']
            bug_id = bug['id']
            history = bug['history']

            for v in bug['comments']:
                vector += str(v) + ' '

            self._c['data'].append(vector)
            self._c['target'].append(developer)

            path = TossingPath(bug_id, developer)
            for assign in history:
                add_dev_name(self._c['target_dict'], assign['email'])
                path.add_path_item((assign['email'], assign['date']))

                tossing_path_collection.append(path)

        self._c['tossing'] = tossing_path_collection

        return self._c
