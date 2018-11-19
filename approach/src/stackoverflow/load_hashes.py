import logging
import time
import sys
import csv

logger = logging.getLogger(__name__)


def load_user_hashes(conn, path='./stackoverflow/hashes.csv'):
    logger.info('loading stackoverflow email hashes from ' + path)
    start_time = time.time()

    try:
        pre = open('./src/stackoverflow/sql/Hashes_pre.sql').read()
    except IOError as e:
        logger.error(e)
        sys.exit(-1)

    with conn.cursor() as cursor:
        try:
            # Pre-processing (dropping/creation of tables)
            logger.info('Pre-processing ...')
            cursor.execute(pre)
            conn.commit()
            logger.info(
                'Pre-processing took {:.1f} seconds'.format(time.time() - start_time))

            with open(path, 'r', encoding="utf-8") as file:

                rows = csv.DictReader(file, delimiter=';')

                # Handle content of the table
                start_time = time.time()
                logger.info('Processing data ...')

                for row in rows:
                    cursor.execute("""INSERT INTO Hashes (Id,name, emailHash, reputation, creationDate)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING ;            
                """, (row['id'], row['name'], row['emailHash'], row['reputation'],
                      row['creationDate']))

                    conn.commit()

                logger.info(
                    'Table processing took {:.1f} seconds'.format(time.time() - start_time))

        except IOError as e:
            logger.error("Could not read from file {}.".format(path))
            logger.error("IOError: {0}".format(e.strerror))
