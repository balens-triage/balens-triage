import logging
import psycopg2.extras
import os

from .get_connection import get_connection

logger = logging.getLogger(__name__)


def noop():
    pass


def here(file):
    return os.path.join(os.path.dirname(__file__), file)


class SaveStorage:

    def __init__(self, db_config, namespace):
        self._ns = namespace

        self._conn = get_connection(db_config)

        with self._conn.cursor() as cursor:
            logger.info('creating tables')
            cursor.execute(open(here('tables.sql'), 'r').read())

            self._conn.commit()

    def __del__(self):
        if self._conn:
            self._conn.close()

    def upsert_bug_and_developer(self, **kwargs):
        with self._conn.cursor() as cursor:
            cursor.execute("""INSERT INTO issues (ns, id, title, summary, assignee, assignee_email, 
cc_list, keywords, status, component, created_at, modified_at, product)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id, ns)
DO UPDATE SET component = EXCLUDED.component, created_at  = EXCLUDED.created_at,
status  = EXCLUDED.status,
modified_at = EXCLUDED.modified_at;            
            """, (self._ns, kwargs['id'], kwargs['title'], kwargs['summary'],
                  kwargs['assignee'], kwargs['assignee_email'],
                  kwargs['cc_list'], kwargs['keywords'], kwargs['status'], kwargs['component'],
                  str(kwargs['created_at']), str(kwargs['modified_at']),
                  kwargs['product']))

            cursor.execute(
                """INSERT INTO developers (ns, dev_user_name, email, real_name, issue_count)
VALUES (%s, %s, %s, %s, 1)
ON CONFLICT (dev_user_name)
  DO UPDATE
    SET issue_count = developers.issue_count + 1""",
                (self._ns, kwargs['assignee_email'], kwargs['assignee_email'],
                 kwargs['assignee']))

            self._conn.commit()

    def upsert_comment(self, **kwargs):
        with self._conn.cursor() as cursor:
            cursor.execute("""INSERT INTO comments (ns, id, bug_id, count, creator, created_at, text)
VALUES (%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (id, ns)
DO UPDATE SET created_at = EXCLUDED.created_at;
            """, (self._ns, kwargs['id'], kwargs['bug_id'], kwargs['count'],
                  kwargs['creator'],
                  str(kwargs['created_at']), kwargs['text']))

            self._conn.commit()

    def upsert_commit_log(self, **kwargs):
        with self._conn.cursor() as cursor:
            cursor.execute("""INSERT INTO commit_logs (ns, commit, message, dev_name, dev_email, bug_ids, created_at)
VALUES (%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (ns, commit)
DO UPDATE SET created_at = EXCLUDED.created_at;            
            """, (self._ns, kwargs['commit'], kwargs['message'], kwargs['dev_name'],
                  kwargs['dev_email'], kwargs['bug_ids'],
                  str(kwargs['created_at'])))

            self._conn.commit()

    def upsert_tossing(self, **kwargs):
        try:
            with self._conn.cursor() as cursor:
                args_str = ','.join(
                    cursor.mogrify("(%s,%s,%s,%s)",
                                   (
                                       self._ns, kwargs['bug_id'], event['email'],
                                       str(event['time'])
                                   )).decode('utf-8') for event in kwargs['events'])

                cursor.execute("""INSERT INTO tossing (ns, bug_id, email, time)
VALUES """ + args_str + """
ON CONFLICT DO NOTHING;
            """)

                self._conn.commit()
        except Exception as e:
            print(e)
            print(kwargs)
            import sys
            sys.exit()

    def upsert_resolution(self, **kwargs):
        try:
            with self._conn.cursor() as cursor:
                args_str = ','.join(
                    cursor.mogrify("(%s,%s,%s,%s)",
                                   (
                                       self._ns, kwargs['bug_id'], event['status'],
                                       str(event['time'])
                                   )).decode('utf-8') for event in kwargs['events'])

                cursor.execute("""INSERT INTO resolution (ns, bug_id, status, time)
    VALUES """ + args_str + """
    ON CONFLICT DO NOTHING;
                """)

                self._conn.commit()
        except Exception as e:
            print(e)
            print(kwargs)
            import sys
            sys.exit()

    def get_existing(self):
        logger.info('loading bug ids ' + self._ns)

        with self._conn.cursor() as cursor:
            cursor.execute("SELECT i.id FROM issues i WHERE i.ns='%s'" % self._ns)

            return cursor.fetchall()

    def get_existing_full(self):
        logger.info('loading bugs ' + self._ns)

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM issues i WHERE i.ns='%s'" % self._ns)

            return cursor.fetchall()

    def update_github(self, id, email):
        with self._conn.cursor() as cursor:
            cursor.execute("UPDATE issues SET assignee_email = %s WHERE id = %s and ns = %s",
                           (email, id, self._ns))

            self._conn.commit()
