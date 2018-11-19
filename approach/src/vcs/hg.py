import csv
import re
import subprocess
from urllib.parse import urlparse
import os.path

import logging

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


fields = ['user', 'date', 'summary']


def get_issue_ids(summary):
    matches = re.findall('Bug([0-9]+)', summary, re.S)

    if len(matches) < 1:
        return None
    else:
        return [int(match) for match in matches]


def get_name_and_email(value):
    m = re.compile('(.*) <(.*)>').match(value)
    if m:
        return m.group(1), m.group(2)
    else:
        return value, value


class MercurialFetcher:

    def __init__(self, url, storage, cache=True):
        self._url = url
        self._storage = storage
        self._domain = urlparse(url).netloc
        self._cache = cache

    def _raw_path(self):
        return './cache/' + self._domain

    def _raw_filename(self):
        return self._domain

    def fetch(self):

        if file_exists(self._raw_path() + '.txt') and self._cache:
            logger.info('Found a log file: ' + self._raw_path() + ' skipping.')
        else:
            logger.info('fetching logs')
            logger.info('This may take some time and can consume gigabytes of disk space!\n')
            self._fetch_logs()

            logger.info('importing commit log to database')

            total = self._import()
            logger.info("{:<15} : {:d}".format("processed issues", total))

    def _fetch_logs(self):
        subprocess.call(['./src/vcs/hg.sh', self._url, self._raw_filename()])

    def _import(self):

        with open(self._raw_path() + '.txt', "r") as raw_file:

            write_record = self._get_writer()

            processed = 0

            record = {}

            for line in raw_file:
                if not line.strip() and record:
                    # empty line is the end of a record
                    record, processed = write_record(record, processed)
                    continue

                field, value = [it.strip() for it in line.split(':', 1)]

                if field == 'changeset':
                    record['commit'] = value

                elif field == 'user':
                    record['dev_name'], record['dev_email'] = get_name_and_email(value)

                elif field == 'date':
                    record['created_at'] = value

                elif field == 'summary':
                    record['message'] = value
                    record['bug_ids'] = get_issue_ids(record['message'])

            if record:
                write_record(record, processed)

            return processed

    def _get_writer(self):
        def write_record(record, processed):
            if record:
                if not record.get('message'):
                    record['message'] = 'N/A'
                    record['bug_ids'] = []

                self._storage.upsert_commit_log(**record)

                processed += 1
                record = {}

            return record, processed

        return write_record
