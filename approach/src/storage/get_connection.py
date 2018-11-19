import sys
import logging

import psycopg2

logger = logging.getLogger(__name__)


def get_connection(db_config):
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' password='%s' host='%s'" % (
                db_config['name'], db_config['user'], db_config['pwd'], db_config['host']
            ))
    except Exception as e:
        logger.error('Failed to connect to database.')
        logger.error(e)
        sys.exit()

    return conn
