import pickle
import time
import logging
import re
from github import Github
from github.GithubObject import NotSet

from src.tools.progress_bar import print_progressbar
from .get_email import get_email
from urllib.parse import urlparse
from .bug import GithubBug

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com/repos/"


def get_owner_and_repo(url):
    path = urlparse(url).path
    [_, owner, repo] = path.split('/')
    return owner, repo


DEFAULT_PER_PAGE = 30


def estimate_total_count(issues):
    url = issues._getLastPageUrl()
    if url:
        return int(re.findall('page=([0-9]+)', url, re.S)[0]) * DEFAULT_PER_PAGE
    else:
        return None


class GitHub:
    def __init__(self, its, max_threads, storage, cache=True):
        self._remote_url = its['url']
        self._token = its['key']
        self._bug_labels = its['bugLabels']
        self._storage = storage
        self._owner, self._repo = get_owner_and_repo(its['url'])
        self._max_threads = max_threads
        self._cache = cache

        self._g = Github(self._token)

    def fetch_all(self):
        repo = self._g.get_repo(self._owner + '/' + self._repo)

        on_bug = self._get_on_bug_handler()

        count = 0
        start_time = time.time()

        labels = [repo.get_label(label) for label in self._bug_labels]

        # get issues starting with the oldest to make the date checkpoints work
        issues = repo.get_issues(labels=labels, sort="updated", state="all",
                                 direction="asc",
                                 since=self._get_checkpoint())

        length = estimate_total_count(issues)

        if length:
            logger.info('Fetching about ' + str(length) + ' GitHub issues from ' + self._remote_url)
        else:
            logger.info('Fetching GitHub issues from ' + self._remote_url)

        for issue in issues:
            if not issue.pull_request:
                count += 1
                print_progressbar(count, length)

                bug = GithubBug(issue)
                on_bug(bug)

        logger.info(
            'Importing {} issues took {:.1f} seconds'.format(count, time.time() - start_time))

    def _get_on_bug_handler(self):
        def on_bug(bug):
            self._set_checkpoint(bug.updated_at())

            history = bug.gethistory()
            resolutions = bug.getresolutions()

            bug_dict = bug.get_transformed()
            assignee_email, fullname = get_email(bug_dict['assignee'], self._token)
            bug_dict['assignee_email'] = assignee_email
            self._storage.upsert_bug_and_developer(**bug_dict)

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

            comments = bug.getcomments()
            for comment in comments:
                self._storage.upsert_comment(**comment)

        return on_bug

    # Checkpoints are necessary, because of api rate limiting
    def _get_checkpoint(self):
        try:
            if not self._cache:
                return None
            return self._unpickle_checkpoint()
        except:
            return NotSet

    def _set_checkpoint(self, date):
        if self._cache:
            self._pickle_checkpoint(date)

    def _pickle_checkpoint(self, date):
        with open(self._get_cache_path(), 'wb') as file:
            pickle.dump(date, file)
            logger.debug('--> created checkpoint cache')

    def _unpickle_checkpoint(self):
        with open(self._get_cache_path(), 'rb') as file:
            date = pickle.load(file)
            logger.debug('--> extracted checkpoint cache: ' + str(date))
            return date

    def _get_cache_path(self):
        return './cache/github_' + self._owner + '_' + self._repo + '_checkpoint.pkl'

    def update(self):
        bugs = self._storage.get_existing_full()

        total_length = len(bugs)
        for index, bug in enumerate(bugs):
            print_progressbar(index, total_length)

            if bug['assignee_email'][0] == '(' and bug['assignee_email'][-1] == ')':
                email = bug['assignee_email'].split(',')[0][1:]
                self._storage.update_github(bug['id'], email)
