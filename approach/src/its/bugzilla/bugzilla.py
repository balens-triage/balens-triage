import threading

import sys
import bugzilla
import logging
from urllib.parse import urlparse

from src.its.bugzilla.bug import BugzillaBug
from src.tools.chunks import n_chunks
from src.tools.progress_bar import print_progressbar
from .bug_fetcher import BugzillaBugFetcher
from .msr import MSRImporter

logger = logging.getLogger(__name__)


class Bugzilla:
    def __init__(self, its, max_threads, storage, cache=True):
        self._projects = its['projects']
        self._remote_url = its['url']
        self._apikey = its['key']
        self._storage = storage
        self._domain = urlparse(self._remote_url).netloc
        self._max_threads = max_threads
        self._cache = cache

        self._bounds = None
        if 'bounds' in its:
            self._bounds = its['bounds']

        # TODO use rest api and add a compabatility option using a wrapper
        self._bzapi = bugzilla.Bugzilla(self._remote_url, api_key=self._apikey)

        if not self._bzapi.logged_in:
            logger.error('--> authentication on ' + self._remote_url + ' failed. Exiting.')
            sys.exit()

    def fetch_all(self):
        on_bug = self._get_on_bug_handler()
        on_tossing = self._get_on_tossing_handler()
        on_resolution = self._get_on_resolution_handler()
        bug_fetcher = BugzillaBugFetcher(self._bzapi, self._max_threads,
                                         self._bounds, self._projects, on_bug)

        msr_fetcher = MSRImporter(self._domain, self._projects, on_tossing, on_resolution)
        bug_fetcher.fetch_bugs()
        # msr_fetcher.fetch()

    def _get_on_bug_handler(self):
        def on_bug(bug):
            comments = bug.getcomments()
            history = bug.gethistory()
            resolutions = bug.getresolutions()
            bug_dict = bug.get_transformed()

            if len(history) > 0:
                self._storage.upsert_tossing(**{
                    'bug_id': bug_dict['id'],
                    'events': history
                })

            if len(resolutions) > 0:
                self._storage.upsert_resolution(**{
                    'bug_id': bug_dict['id'],
                    'events': resolutions
                })

            self._storage.upsert_bug_and_developer(**bug_dict)

            for comment in comments:
                comment['created_at'] = comment['creation_time']
                self._storage.upsert_comment(**comment)

        return on_bug

    def _get_on_tossing_handler(self):
        def on_tossing(assignment):
            self._storage.upsert_tossing(**assignment)

        return on_tossing

    def _get_on_resolution_handler(self):
        def on_resolution(resolution):
            self._storage.upsert_resolution(**resolution)

        return on_resolution

    def update(self):
        bugs = self._storage.get_existing()
        logger.info('updating %d bugs using %d threads' % (len(bugs), self._max_threads))

        on_bug_handler = self._get_on_bug_handler()
        bug_writer = UpdateBugWriter(len(bugs), on_bug_handler)

        bug_lists = n_chunks(bugs, self._max_threads)
        threads = [self._get_thread(bugs, bug_writer) for bugs in bug_lists]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _get_thread(self, bugs, bug_writer):
        return threading.Thread(target=self._update_bugs, args=(bugs, bug_writer))

    def _update_bugs(self, bugs, bug_writer):
        for index, bug in enumerate(bugs):
            bug = BugzillaBug(bzapi=self._bzapi, bug_id=bug[0])
            if bug.exists():
                bug_writer.write(bug)


class UpdateBugWriter:

    def __init__(self, total, callback):
        self._total = total
        self._callback = callback
        self._processed = 0

    def write(self, bug):
        self._processed += 1
        print_progressbar(self._processed, self._total)
        self._callback(bug)
