import os.path
import logging
from enum import Enum
from .date_bounds import BugzillaDateBoundFinder
from .max_bounds import BugzillaMaxBoundFinder
from .projects_bounds import BugzillaProjectBoundFinder

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


class Strategies(Enum):
    FIND_BOUNDS = 1
    DO_NOTHING = 2
    FIND_DATE_BOUNDS = 3


class LazyFinder:
    def __init__(self, bounds):
        self._bounds = bounds

    def get_bounds(self):
        return self._bounds[0], self._bounds[1]


class BugzillaBoundFinder:
    def __init__(self, bzapi, bounds, projects):
        self._bzapi = bzapi
        self._projects = projects
        self._strategy = self._parse_bounds(bounds)
        self._lower, self._upper = None, None

    def _parse_bounds(self, bounds):
        if not bounds:
            return BugzillaMaxBoundFinder(self._bzapi, bounds)
        elif isinstance(bounds[0], str) and isinstance(bounds[1], str):
            if len(self._projects) > 0:
                return BugzillaProjectBoundFinder(self._bzapi, bounds, self._projects)
            else:
                return BugzillaDateBoundFinder(self._bzapi, bounds)
        elif isinstance(bounds[0], int) and isinstance(bounds[1], int):
            return LazyFinder(bounds)
        else:
            raise Exception('Invalid bounds supplied to BugzillaBoundFinder.')

    def get_bounds(self):
        if self._lower and self._upper:
            return self._lower and self._upper
        else:
            return self.fetch_bounds()

    def fetch_bounds(self):
        self._lower, self._upper = self._strategy.get_bounds()
        return self._lower, self._upper
