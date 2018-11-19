import logging
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
import numpy as np

from src.tools.decorators.cached import mem_cached
from src.tools.tossing.developer_activity import get_fix_times_per_developer

logger = logging.getLogger(__name__)


def fix_time_estimate(events_per_developer, vectorizer, test_acc=False):
    clf_per_developer = {}
    bug_ft = {}

    acc_scores = []

    for developer, events in events_per_developer.items():
        bug_vectors = vectorizer([event[1] for event in events])
        bug_ids = [event[2] for event in events]
        fix_times = [event[0] for event in events]

        if len(fix_times) > 9:
            clf = LinearSVR(C=1000)

            x_train, x_test, y_train, y_test = train_test_split(bug_vectors, fix_times,
                                                                test_size=0.2,
                                                                random_state=42)

            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            clf.fit(bug_vectors, fix_times)
            if score > 0.0:
                acc_scores.append(clf.score(x_test, y_test))
                clf_per_developer[developer] = clf
        for index, bug_id in enumerate(bug_ids):
            bug_ft[bug_id] = fix_times[index]

    logger.info("%d out of %d developers covered by fix time estimation" % (
        len(acc_scores), len(events_per_developer.items())))

    if test_acc:
        print("mean developer fix time r^2 %.2f (+/- %.2f)" % (
            np.mean(acc_scores), np.std(acc_scores)))
        import sys
        sys.exit()
    return clf_per_developer, bug_ft


class FixTimeEstimator:

    def __init__(self, data_rows, vectorizer):
        events_per_developer = get_fix_times_per_developer(data_rows)
        self._vectorizer = vectorizer
        self._estimators, self._bug_ft = fix_time_estimate(events_per_developer, vectorizer)
        del events_per_developer

    def predict(self, developer_name, bug_vector):
        if developer_name not in self._estimators:
            # We cannot make an estimation
            return float('inf')
        else:
            vector = self._vectorizer([bug_vector])
            return self._estimators[developer_name].predict(vector)[0]

    @mem_cached()
    def get_prediction_times(self, bug_vector):
        predictions = []

        for developer, estimator in self._estimators.items():
            predictions.append((developer, self.predict(developer, bug_vector)))

        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def actual_ft(self, bug_id):
        if bug_id not in self._bug_ft:
            raise Exception(str(bug_id) + 'not found')
        else:
            return self._bug_ft[bug_id]
