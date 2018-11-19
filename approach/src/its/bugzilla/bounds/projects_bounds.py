import os.path
from datetime import datetime


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


dateformat = '%Y-%m-%d'


def parse_bounds(bounds):
    if isinstance(bounds[0], str) and isinstance(bounds[1], str):
        return [datetime.strptime(bounds[0], dateformat), datetime.strptime(bounds[1], dateformat)]
    else:
        raise Exception('Invalid bounds supplied to BugzillaBoundFinder.')


class BugzillaProjectBoundFinder:
    def __init__(self, bzapi, bounds, projects):
        self._projects = projects
        self._bzapi = bzapi
        self._bounds = parse_bounds(bounds)

    def get_bounds(self):
        return self._find_lower_bound(), self._find_upper_bound()

    def _find_upper_bound(self):
        return self._fetch_by_date(self._bounds[1])

    def _find_lower_bound(self):
        return self._fetch_by_date(self._bounds[0])

    def _upper(self):
        return self._bounds[1]

    def _lower(self):
        return self._bounds[0]

    def _fetch_by_date(self, earliest_date):
        date_string = earliest_date.strftime("%Y-%m-%d")

        query = self._bzapi.build_query()
        query["creation_time"] = date_string
        query["product"] = self._projects
        query["limit"] = 1

        bugs = self._bzapi.query(query)

        if len(bugs) < 1:
            raise Exception('no bugs found after ' + date_string)
        else:
            return bugs[0].id
