import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import tree

logger = logging.getLogger(__name__)


class SimpleDecisionTree:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._container = container

    def train(self):
        logger.info("training using Simple")

        X = self.load()

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                X['data'], X['target'], range(0, len(X['data'])), test_size=0.2, random_state=101)

        logger.info("training data set size: " + str(len(x_train)))
        logger.info("testing  data set size: " + str(len(x_test)))

        clf = Pipeline([('tfidf', TfidfVectorizer(lowercase=True,
                                                  min_df=15,
                                                  stop_words='english',
                                                  analyzer='word')),
                        ('clf', tree.DecisionTreeClassifier())])

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
