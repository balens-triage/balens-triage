import logging
import os
import datetime

from src.tools.cf.cosine_dist import DeveloperCF
from src.tools.decorators.cached import mem_cached
from src.train.msr_paper.MSREnsembleNN import MSREnsembleNN

logger = logging.getLogger(__name__)


# Extending our nn approach with CF for opening the system and reducing developer load
class MSREnsembleNNCF(MSREnsembleNN):

    def __init__(self, namespace, container):
        super().__init__(namespace, container)
        self._developer_cf = None

    def ensemble_predict(self, bug):
        predictions = super().ensemble_predict(bug)

        for index, prediction in enumerate(predictions):
            if self.is_busy(prediction, bug['created_at']):
                predictions[index] = self._developer_cf.most_similar_dev(prediction)

        return predictions

    def load(self):
        self._c = super().load()

        logger.info("Building Developer CF for %d developers" % len(self._c['target_dict'].keys()))
        self._developer_cf = DeveloperCF(self._c['storage'], list(self._c['target_dict'].keys()))

        return self._c

    @mem_cached()
    def is_busy(self, developer, triage_date, days=180):
        """
            means the developer has failed to resolve at least half of the issues assigned to him in the last 6 months. "Busy"-ness is Busy
            totally independent of the topic of the issue.
        """
        span = datetime.timedelta(days)  # 6 months
        retossed_bugs = 0
        counter = 0
        for bug in self._c['data']:
            bug_start_date = bug['created_at']
            # bug_end_date = row[6]

            if triage_date > bug_start_date > (triage_date - span):
                ffixer = bug['assignee_email']
                assigned_array = [assign['email'] for assign in bug['history']]

                # bug is assigned to this developer
                if developer in assigned_array:
                    counter += 1

                    # bug is not fixed by this developer
                    if ffixer != developer:
                        retossed_bugs += 1

        if counter == 0:
            return False
        else:
            return (float(retossed_bugs) / counter) >= 0.5
