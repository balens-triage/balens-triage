import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.base import Trainer

logger = logging.getLogger(__name__)


class SimpleSVM(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._container = container

    def train(self):
        logger.info("Training using Simple")

        X = self.load()

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                X['data'], X['target'], range(0, len(X['data'])), test_size=0.2, random_state=101)

        logger.info("training data set size: " + str(len(x_train)))
        logger.info("testing  data set size: " + str(len(x_test)))

        clf = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                  stop_words='english',
                                                  analyzer='word')),
                        ('clf',
                         SVC(kernel="rbf", probability=True, C=100.0, gamma=0.1))])

        clf.fit(x_train, y_train)

        prediction = clf.predict(x_test)

        self._container['reports'].append({
            'trainer': self.__class__.__name__,
            'pred_acc': np.mean(prediction == y_test)
        })

        print("prediction accuracy on " + self.__class__.__name__ + ": ")
        print(np.mean(prediction == y_test))

        return self._container

    def load(self):
        return self._container
