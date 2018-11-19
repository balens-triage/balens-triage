import logging

from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.utils.extmath import density
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, \
    SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC

from src.base import Trainer
from src.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SimpleOptim(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._container = container

    def train(self):
        # TODO not relevant for paper but important

        X = self.load()

        x_train, x_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(
                X['data'], X['target'], range(0, len(X['data'])), test_size=0.2, random_state=42)
        print('data loaded')

        # order of labels in `target_names` can be different from `categories`
        target_names = X['target_names']

        def size_mb(docs):
            return sum(len(s.encode('utf-8')) for s in docs) / 1e6

        data_train_size_mb = size_mb(x_train)
        data_test_size_mb = size_mb(x_test)

        print("%d documents - %0.3fMB (training set)" % (
            len(x_train), data_train_size_mb))
        print("%d documents - %0.3fMB (test set)" % (
            len(x_test), data_test_size_mb))
        print("%d categories" % len(target_names))
        print()

        print("Extracting features from the training data using a sparse vectorizer")
        t0 = time()
        if False:
            vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                           n_features=2 ** 16)
            X_train = vectorizer.transform(x_train)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
            X_train = vectorizer.fit_transform(x_train)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_train.shape)
        print()

        print("Extracting features from the test data using the same vectorizer")
        t0 = time()
        X_test = vectorizer.transform(x_test)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()

        # mapping from integer feature name to original token string
        if False:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        if feature_names:
            feature_names = np.asarray(feature_names)

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."

        # #############################################################################
        # Benchmark classifiers
        def benchmark(clf):
            print('_' * 80)
            print("Training: ")
            print(clf)
            t0 = time()
            clf.fit(X_train, y_train)
            train_time = time() - t0
            print("train time: %0.3fs" % train_time)

            t0 = time()
            pred = clf.predict(X_test)
            test_time = time() - t0
            print("test time:  %0.3fs" % test_time)

            score = metrics.accuracy_score(y_test, pred)
            print("accuracy:   %0.3f" % score)

            if hasattr(clf, 'coef_'):
                print("dimensionality: %d" % clf.coef_.shape[1])
                print("density: %f" % density(clf.coef_))

                if False and feature_names is not None:
                    print("top 10 keywords per class:")
                    for i, label in enumerate(target_names):
                        top10 = np.argsort(clf.coef_[i])[-10:]
                        print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                print()

            print("classification report:")
            print(metrics.classification_report(y_test, pred,
                                                target_names=target_names))

            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

            print()
            clf_descr = str(clf).split('(')[0]
            return clf_descr, score, train_time, test_time

        results = []
        for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
            print('=' * 80)
            print(name)
            results.append(benchmark(clf))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                               tol=1e-3)))

            # Train SGD model
            results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                   penalty=penalty)))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(benchmark(MultinomialNB(alpha=.01)))
        results.append(benchmark(BernoulliNB(alpha=.01)))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                            tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))])))

        # make some plots

        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()

        return self._container

    def load(self):
        return self._container
