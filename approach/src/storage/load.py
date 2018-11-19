import logging
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import os
import re

from datetime import datetime

from src.tools.decorators.cached import cached
from .get_connection import get_connection

logger = logging.getLogger(__name__)


def noop():
    pass


def here(file):
    return os.path.join(os.path.dirname(__file__), file)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def transform_hist_array(array_string):
    result = []
    for split in re.findall(r'(\"\(.*?\)\")', array_string):
        tpl = split[2:-2]
        [ns, bug_id, email, date] = tpl.split(',')

        date = date[2:-2]  # remove \\"

        result.append({
            'ns': ns,
            'bug_id': bug_id,
            'email': email,
            'date': datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        })

    return sorted(result, key=lambda x: x['date'], reverse=True)


def transform_resolution_array(array_string):
    result = []
    for split in re.findall(r'(\"\(.*?\)\")', array_string):
        tpl = split[2:-2]
        [ns, bug_id, status, date] = tpl.split(',')

        date = date[2:-2]  # remove \\"

        result.append({
            'ns': ns,
            'bug_id': bug_id,
            'status': status,
            'time': datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        })

    return sorted(result, key=lambda x: x['time'], reverse=True)


class LoadStorage:

    def __init__(self, config, namespace, excluded_emails=None):
        self._ns = namespace
        self._data_limit = config['data_limit']

        psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)

        db_config = config['db']
        self._conn = get_connection(db_config)

        if excluded_emails is None:
            excluded_emails = []

        self._excluded_emails = excluded_emails

    def __del__(self):
        if self._conn:
            self._conn.close()

    @property
    def _limit(self):
        if self._data_limit:
            return " LIMIT " + str(self._data_limit)
        else:
            return ""

    @property
    def _excluded(self):
        if len(self._excluded_emails) > 0:
            return " AND i.assignee_email NOT IN (" + ",".join(self._excluded_emails) + ") "
        else:
            return ""

    def _map(self, results):
        if 'github' in self._ns:
            def map_issue(issue):
                issue['assignee_email'] = issue['assignee']
                return issue

            return list(map(map_issue, results))
        else:
            return results

    @cached('bugs_and_comments.pkl')
    def load_bugs_and_comments(self):
        logger.info('loading bugs and comments from namespace ' + self._ns)

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""SELECT
i.id,
i.assignee_email,
i.summary,
i.title,
i.status,
i.keywords,
i.assignee,
i.modified_at,
array_agg(distinct c.text) as comments
FROM issues i
LEFT JOIN comments c ON c.bug_id = i.id
WHERE i.assignee_email is not null
AND i.ns=%s AND c.ns=%s """ + self._excluded + """
AND i.status = 'VERIFIED FIXED'
GROUP BY i.id, i.ns""" + self._limit, (self._ns, self._ns))

            results = cursor.fetchall()

            return results

    @cached('all_bugs_and_comments.pkl')
    def load_all_bugs_and_comments(self):
        logger.info('loading bugs and comments from namespace ' + self._ns)

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""SELECT
    i.id,
    i.assignee_email,
    i.summary,
    i.title,
    i.status,
    i.keywords,
    i.assignee,
    i.modified_at,
    i.created_at,
    array_agg(distinct c.text) as comments
    FROM issues i
    LEFT JOIN comments c ON c.bug_id = i.id
    WHERE i.assignee_email is not null AND i.status = 'VERIFIED FIXED'
    AND i.ns=%s AND c.ns=%s """ + self._excluded + """
    GROUP BY i.id, i.ns
            """ + self._limit, (self._ns, self._ns))

            results = cursor.fetchall()

            return results

    @cached('bug_and_history.pkl')
    def load_bugs_and_history(self):
        logger.info('loading bugs and their history from namespace ' + self._ns)

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""SELECT
i.id,
i.assignee_email,
i.summary,
i.title,
i.status,
i.keywords,
i.assignee,
i.component,
i.modified_at,
i.created_at,
array_agg(distinct c.text) as comments,
array_agg(distinct t) as history,
array_agg(distinct r) as resolution
FROM issues i
LEFT JOIN comments c ON c.bug_id = i.id
LEFT JOIN tossing t ON t.bug_id = i.id
LEFT JOIN resolution r ON r.bug_id = i.id
WHERE i.assignee_email is not null
AND t.email != 'NA'
AND i.ns=%s AND c.ns=%s AND t.ns=%s  """ + self._excluded + """
GROUP BY i.id, i.ns ORDER BY i.modified_at
""" + self._limit, (self._ns, self._ns, self._ns))

            results = cursor.fetchall()

            for result in results:
                result['history'] = transform_hist_array(result['history'])
                result['resolution'] = transform_resolution_array(result['resolution'])

            results = self._map(results)

            return results

    @cached('commit_logs.pkl')
    def load_developer_content(self, developers):
        logger.info(
            'loading commit logs for %d developers from namespace %s' % (len(developers), self._ns))

        if 'github' in self._ns:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""SELECT
       log.dev_email as email,
       array_agg(log.message) as messages,
       array_agg(p.body) as posts
FROM commit_logs log, issues i, posts p, hashes h
WHERE i.assignee_email != 'N/A' AND md5(i.assignee_email) = h.emailhash and
      p.id = h.id AND
      log.ns=%s AND log.dev_email = i.assignee_email AND i.assignee IN %s
    GROUP BY log.dev_email, log.ns
""" + self._limit, (self._ns, developers))

                return cursor.fetchall()
        else:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""SELECT
       log.dev_email as email,
       array_agg(log.message) as messages,
       array_agg(p.body) as posts
FROM commit_logs log, issues i, posts p, hashes h
WHERE md5(log.dev_email) = h.emailhash and
      p.id = h.id AND
    log.ns=%s AND log.dev_email IN %s 
    GROUP BY log.dev_email, log.ns
        """ + self._limit, (self._ns, developers))

                return cursor.fetchall()
