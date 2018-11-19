import logging
from sklearn.datasets.base import Bunch

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)


class SimpleTF:
    def __init__(self, namespace, data_rows, developer_ids):
        self._ns = namespace
        self._data_rows = data_rows
        self._developer_ids = developer_ids

    def train(self):
        logger.info("training using Simple")
        # TODO

    def load(self):
        X = Bunch()
        X['data'] = []
        X['target'] = []
        X['target_names'] = list(set(self._developer_ids))

        for bug in self._data_rows:
            developer = bug['assignee_email']
            vector = bug['summary']

            for v in bug['comments']:
                vector += str(v) + ' '

            X['data'].append(vector)
            X['target'].append(developer)

        return X
