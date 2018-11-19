import logging
import os.path
from datetime import datetime
from ..bug import BugzillaBug


logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


dateformat = '%Y-%m-%d'


def parse_bounds(bounds):
    if isinstance(bounds[0], str) and isinstance(bounds[1], str):
        return [datetime.strptime(bounds[0], dateformat), datetime.strptime(bounds[1], dateformat)]
    else:
        raise Exception('Invalid bounds supplied to BugzillaBoundFinder.')


"""
This class determines the maximum size of a bugzilla repository
"""


class BugzillaMaxBoundFinder:
    def __init__(self, bzapi, bounds):
        self._bzapi = bzapi
        self._bounds = bounds

    def get_bounds(self):
        return 0, self._find_bound()

    def _find_bound(self):
        lower, upper = self._find_maximum_upper()
        logging.debug('--> binary search for upper bound [' + str(lower) + ', ' + str(upper) + ']')

        left = lower
        right = upper

        while left <= right:
            i = (left + right) // 2

            bug = self._get_bug(i)
            next_bug = self._get_bug(i + 1)

            if self._guess_correct(bug, next_bug):
                return i
            elif self._guess_was_low(bug, next_bug):
                left = i + 1
            else:
                right = i - 1

    def _guess_correct(self, bug, next_bug):
        return bug.exists() and not next_bug.exists()

    def _guess_was_low(self, bug, next_bug):
        return bug.exists() and next_bug.exists()

    def _find_maximum_upper(self):
        logging.debug('\n--> Probing maximum upper bound ...')
        lower, upper = 0, 10000

        while True:
            bug = self._get_bug(upper)

            if bug.exists():
                lower = upper
                upper = 2 * upper
            else:
                return lower, upper

    def _get_bug(self, bug_id):
        return BugzillaBug(bzapi=self._bzapi, bug_id=bug_id)
