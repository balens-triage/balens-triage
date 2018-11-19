import logging
import threading
import pickle
import math
import os.path

from src.tools.progress_bar import print_progressbar
from .bounds import BugzillaBoundFinder
from .bug import BugzillaBug

from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def is_string_list(a_list):
    return filter(lambda x: isinstance(x, str), a_list) == 0


class BugzillaBugFetcher:
    # empirical constant
    _bugzilla_chunk_size = 1000

    def __init__(self, bzapi, max_threads, bounds, projects, on_bug, cache=True):
        self._on_bug = on_bug
        self._max_threads = max_threads
        self._bounds = bounds
        self._cache = cache
        self._lower_date_bound = None
        self._projects = projects
        self._use_projects = len(projects) > 0

        self._bzapi = bzapi
        self._domain = urlparse(bzapi.url).netloc

        self._bound_finder = BugzillaBoundFinder(self._bzapi, bounds, projects)

    def fetch_bugs(self):
        logger.info('Fetching Bugzilla bugs ...')

        lower, upper = self._get_bounds()

        if lower > upper:
            logger.info('Found cache that says bugs already fetched. Skipping.')
            return

        total = upper - lower

        x = {'failed': 0, 'completed': 0}

        def write(bug):
            if bug == 'failed':
                x['failed'] += 1
            elif bug:
                self._on_bug(bug)

            x['completed'] += 1
            print_progressbar(x['completed'], total - x['failed'])

        print_progressbar(0, total)
        self._run(lower, upper, write)
        logger.info('\n--> completed with ' + str(x['failed']) + '  failed bug fetches: ')

    def _get_bounds(self):
        try:
            if not self._cache:
                raise Exception()
            lower, upper = self._unpickle_bounds()
        except:
            lower, upper = self._bound_finder.get_bounds()

        if self._use_projects and 0 < lower < upper:
            self._lower_date_bound = self._bzapi.getbug(lower).creation_time

        if lower > upper and not self._cache:
            lower, upper = self._bound_finder.get_bounds()

        self._pickle_bounds(lower, upper)
        return lower, upper

    def _pickle_bounds(self, lower, upper):
        with open(self._get_cache_path(), 'wb') as file:
            pickle.dump([lower, upper], file)
            logger.debug('--> created bounds cache')

    def _unpickle_bounds(self):
        with open(self._get_cache_path(), 'rb') as file:
            lower, upper = pickle.load(file)
            logger.debug('--> extracted bounds cache: [%d, %d]' % (lower, upper))
            return lower, upper

    def _get_cache_path(self):
        return './cache/' + self._domain + '_bounds.pkl'

    def _run(self, lower, upper, write):
        try:
            if self._use_projects:
                total = upper - lower

                # split the total number into batches that can be run in threads and respect
                # the bugzilla chunk size
                batch_size = self._max_threads * self._bugzilla_chunk_size
                batches = math.ceil(total / batch_size)
                logger.info('--> running batched fetches over project(s) %s using %d thread(s)' % (
                    ','.join(self._projects), self._max_threads
                ))

                for batch in range(0, int(batches)):
                    self._run_helper(lower + batch * batch_size,
                                     lower + (batch + 1) * batch_size, write)
                    self._pickle_bounds(lower + (batch + 1) * batch_size, upper)
            else:
                return self._run_helper(lower, upper, write)
        except (KeyboardInterrupt, SystemExit):
            logger.warning('\nReceived keyboard interrupt, quitting threads!\n')

    def _run_helper(self, lower, upper, write):
        threads = self._get_threads(lower, upper, write)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _print_bound_info(self, lower, upper):
        if self._use_projects:
            logger.debug('fetching bugs from %s in approx. id interval [%d, %d] %d thread(s)' % (
                ','.join(self._projects), lower, upper, self._max_threads))
        else:
            logger.debug('fetching %d bugs in id interval [%d, %d] %d thread(s)' % (
                upper - lower, lower, upper, self._max_threads))

    def _get_threads(self, lower, upper, write):
        if self._use_projects and self._using_date_bounds():
            # we are not fetching bug ids, but offsets within a project id range
            lower = 0

        chunk_size = math.ceil((upper - lower) / self._max_threads)

        return list(map(lambda x: self._get_thread(lower + x * chunk_size,
                                                   lower + (x + 1) * chunk_size,
                                                   write),
                        range(0, self._max_threads)))

    def _using_date_bounds(self):
        return self._bounds and len(self._bounds) > 1 and is_string_list(self._bounds)

    def _get_thread(self, lower, upper, write):
        return threading.Thread(target=self._fetch, args=(lower, upper, write))

    def _fetch(self, lower, upper, write):
        if self._use_projects:
            query = self._get_query(lower, upper)
            try:
                bugs = self._bzapi.query(query)
            except Exception as e:
                print(e)
                return write('failed')

            for bug in bugs:

                if self._using_date_bounds() and bug.id > upper:
                    break

                if bug.product in self._projects:
                    write(BugzillaBug(bug=bug))
        else:
            for bug_id in range(lower, upper):
                bug = self.get_bug(bug_id)
                if bug.exists():
                    write(bug)
                else:
                    write('failed')

    def _get_query(self, lower, upper):
        query = self._bzapi.build_query()

        if self._lower_date_bound:
            query["creation_time"] = self._lower_date_bound

        query["product"] = self._projects
        query["limit"] = upper - lower
        query["offset"] = lower
        return query

    def get_bug(self, bug_id):
        return BugzillaBug(bzapi=self._bzapi, bug_id=bug_id)
