import sys
import time
from ..tools import batch, parse

import logging

logger = logging.getLogger(__name__)

# Special rules needed for certain tables (esp. for old database dumps)
special_rules = {
    ('Posts', 'ViewCount'): "NULLIF(%(ViewCount)s, '')::int"
}


def _make_def_values(keys):
    """Returns a dictionary containing None for all keys."""
    return dict(((k, None) for k in keys))


def _create_mogrification_template(table, keys):
    """Return the template string for mogrification for the given keys."""
    return ('(' +
            ', '.join(
                ['%(' + k + ')s' if (table, k) not in special_rules else special_rules[table, k]
                 for k in keys
                 ]
            ) +
            ')'
            )


def _create_cmd_tuple(cursor, keys, templ, attribs):
    """Use the cursor to mogrify a tuple of data.
    The passed data in `attribs` is augmented with default data (NULLs) and the
    order of data in the tuple is the same as in the list of `keys`. The
    `cursor` is used toe mogrify the data and the `templ` is the template used
    for the mogrification.
    """
    defs = _make_def_values(keys)
    defs.update(attribs)
    return cursor.mogrify(templ, defs)


def load_table(table, data_file, conn):
    keys = get_keys_from_table(table)

    """Handle the table including the post/pre processing."""
    db_file = data_file if data_file is not None else table + '.xml'
    tmpl = _create_mogrification_template(table, keys)
    start_time = time.time()

    try:
        pre = open('./src/stackoverflow/sql/' + table + '_pre.sql').read()
        post = open('./src/stackoverflow/sql/' + table + '_post.sql').read()
    except IOError as e:
        logger.error(e)
        sys.exit(-1)

    with conn.cursor() as cur:
        try:
            with open(db_file, 'rb') as xml:
                # Pre-processing (dropping/creation of tables)
                logger.info('Pre-processing ...')
                if pre != '':
                    cur.execute(pre)
                    conn.commit()
                logger.info(
                    'Pre-processing took {:.1f} seconds'.format(time.time() - start_time))

                # Handle content of the table
                start_time = time.time()
                logger.info('Processing data ...')

                for rows in batch(parse(xml), 500):
                    values_str = ',\n'.join(
                        [_create_cmd_tuple(cur, keys, tmpl, row_attribs.attrib).decode('utf-8')
                         for row_attribs in rows
                         ]
                    )

                    if len(values_str) > 0:
                        cmd = 'INSERT INTO ' + table + \
                              ' VALUES\n' + values_str + ';'
                        cur.execute(cmd)
                        conn.commit()

                logger.info(
                    'Table processing took {:.1f} seconds'.format(time.time() - start_time))

                # Post-processing (creation of indexes)
                start_time = time.time()
                logger.info('Post processing ...')
                if post != '':
                    cur.execute(post)
                    conn.commit()
                logger.info(
                    'Post processing took {} seconds'.format(time.time() - start_time))

        except IOError as e:
            logger.error("Could not read from file {}.".format(db_file))
            logger.error("IOError: {0}".format(e.strerror))


def get_keys_from_table(table):
    if table == 'Users':
        return [
            'Id'
            , 'Reputation'
            , 'CreationDate'
            , 'DisplayName'
            , 'LastAccessDate'
            , 'WebsiteUrl'
            , 'Location'
            , 'AboutMe'
            , 'Views'
            , 'UpVotes'
            , 'DownVotes'
            , 'ProfileImageUrl'
            , 'Age'
            , 'AccountId'
        ]
    elif table == 'Badges':
        return [
            'Id'
            , 'UserId'
            , 'Name'
            , 'Date'
        ]
    elif table == 'PostLinks':
        return [
            'Id'
            , 'CreationDate'
            , 'PostId'
            , 'RelatedPostId'
            , 'LinkTypeId'
        ]
    elif table == 'Comments':
        return [
            'Id'
            , 'PostId'
            , 'Score'
            , 'Text'
            , 'CreationDate'
            , 'UserId'
        ]
    elif table == 'Votes':
        return [
            'Id'
            , 'PostId'
            , 'VoteTypeId'
            , 'UserId'
            , 'CreationDate'
            , 'BountyAmount'
        ]
    elif table == 'Posts':
        return [
            'Id'
            , 'PostTypeId'
            , 'AcceptedAnswerId'
            , 'ParentId'
            , 'CreationDate'
            , 'Score'
            , 'ViewCount'
            , 'Body'
            , 'OwnerUserId'
            , 'LastEditorUserId'
            , 'LastEditorDisplayName'
            , 'LastEditDate'
            , 'LastActivityDate'
            , 'Title'
            , 'Tags'
            , 'AnswerCount'
            , 'CommentCount'
            , 'FavoriteCount'
            , 'ClosedDate'
            , 'CommunityOwnedDate'
        ]
    elif table == 'Tags':
        return [
            'Id'
            , 'TagName'
            , 'Count'
            , 'ExcerptPostId'
            , 'WikiPostId'
        ]
    elif table == 'PostHistory':
        return [
            'Id',
            'PostHistoryTypeId',
            'PostId',
            'RevisionGUID',
            'CreationDate',
            'UserId',
            'Text'
        ]
    elif table == 'Comments':
        return [
            'Id',
            'PostId',
            'Score',
            'Text',
            'CreationDate',
            'UserId',
        ]
