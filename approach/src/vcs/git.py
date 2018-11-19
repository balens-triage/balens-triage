import re
import subprocess
from urllib.parse import urlparse
import os.path

import logging

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def get_issue_ids(summary):
    matches = re.findall('#([0-9]+)', summary, re.S)

    if len(matches) < 1:
        return None
    else:
        return [int(match) for match in matches]


class GitFetcher:

    def __init__(self, url, storage, namespace, cache=True):
        self._url = url
        self._ns = namespace
        self._storage = storage
        self._domain = urlparse(url).netloc
        self._cache = cache

    def _raw_path(self):
        return './cache/' + self._raw_filename()

    def _raw_filename(self):
        return self._ns + '_git'

    def fetch(self):

        if file_exists(self._raw_path() + '.txt') and self._cache:
            logger.info('Found a log file: ' + self._raw_path())
        else:
            logger.info('fetching logs')
            logger.info('This may take some time and can consume gigabytes of disk space!\n')
            self._fetch_logs()

        logger.info('importing commit log to database')

        total = self._import()
        logger.info("{:<15} : {:d}".format("processed", total))

    def _fetch_logs(self):
        subprocess.call(['./src/vcs/git.sh', self._url, self._raw_filename()])

    def _import(self):

        with open(self._raw_path() + '.txt', "r", encoding='utf-8') as raw_file:
            processed = 0

            record = {}

            write_record = self.get_writer()

            # parsing logic taken from here:
            # https://github.com/johnkchiu/GitLogParser/blob/master/gitLogParser.py

            for line in raw_file:

                if line == '' or line == '\n':
                    pass

                elif bool(re.match('commit', line, re.IGNORECASE)):
                    if len(record) != 0:
                        record, processed = write_record(record, processed)

                    record = {'commit': re.match('commit (.*)', line, re.IGNORECASE).group(1)}

                elif bool(re.match('merge:', line, re.IGNORECASE)):
                    pass

                elif bool(re.match('author:', line, re.IGNORECASE)):
                    m = re.compile('Author: (.*) <(.*)>').match(line)
                    record['dev_name'] = m.group(1)
                    record['dev_email'] = m.group(2)

                elif bool(re.match('date:', line, re.IGNORECASE)):
                    m = re.findall('Date: (.*)', line)
                    record['created_at'] = m[0].strip()
                    pass

                elif bool(re.match('    ', line, re.IGNORECASE)):
                    if record.get('message') is None:
                        record['message'] = line.strip()
                    else:
                        record['message'] += line.strip()

                else:
                    logging.warning('Unexpected Line: ' + line)

            if record:
                # handle last record
                write_record(record, processed)

            return processed

    def get_writer(self):
        def write_record(record, processed):
            if record:
                if 'message' in record:
                    issues_ids = get_issue_ids(record['message'])
                    record['bug_ids'] = issues_ids
                else:
                    record['bug_ids'] = []
                    record['message'] = ''

                self._storage.upsert_commit_log(**record)

                processed += 1
                record = {}

            return record, processed

        return write_record
