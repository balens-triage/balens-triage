import logging
import tarfile
from xml.etree import ElementTree

import time
import os.path
import requests
import datetime

from ...tools import batch, parse

logger = logging.getLogger(__name__)


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def dir_exists(path):
    return os.path.isdir(path) and os.path.exists(path)


def is_string_list(a_list):
    return filter(lambda x: isinstance(x, str), a_list) == 0


MOZ_SUPPORTED_PROJECTS = ['Bugzilla', 'Core', 'Firefox', 'Thunderbird']
ECLIPSE_SUPPORTED_PROJECTS = ['CDT', 'JDT', 'PDE', 'Platform']

ALL_SUPPORTED_PROJECTS = MOZ_SUPPORTED_PROJECTS + ECLIPSE_SUPPORTED_PROJECTS


def projects_valid(projects):
    for project in projects:
        if project not in ALL_SUPPORTED_PROJECTS:
            return False
    else:
        return True


def here(file):
    return os.path.join(os.path.dirname(__file__), file)


ECLIPSE_URL = 'https://github.com/ansymo/msr2013-bug_dataset/raw/master/data/eclipse.tar.gz'
MOZILLA_URL = 'https://github.com/ansymo/msr2013-bug_dataset/raw/master/data/mozilla.tar.gz'


def download_file(url, local_target):
    r = requests.get(url, stream=True)
    with open(local_target, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


assigned_to_naming = {
    'when': 'time',
    'what': 'email'
}

resolution_naming = {
    'when': 'time',
    'what': 'status'
}


def get_update_events(elem, name_mapping=assigned_to_naming):
    updates = []
    when = name_mapping['when']
    what = name_mapping['what']

    for update_elem in elem.getchildren():
        update = {}

        children = update_elem.getchildren()

        for child in children:
            if child.tag == 'when':
                update[when] = datetime.datetime.fromtimestamp(int(child.text))
            elif child.tag == 'what':
                update[what] = child.text

        if not update.get('email'):
            update[what] = 'NA'

        updates.append(update)

    return updates


class MSRImporter:

    def __init__(self, domain, projects, on_item, on_resolution, max_threads=1, cache=True):
        self._on_item = on_item
        self._on_resolution = on_resolution
        self._max_threads = max_threads
        self._cache = cache

        if len(projects) < 1 or not projects_valid(projects):
            raise Exception("""The MSR defect data set only contains the following projects: %s
            Getting the assignment history for any other project is currently not supported.             
            """ % ", ".join(ALL_SUPPORTED_PROJECTS))

        self._projects = projects
        self._domain = domain

    def fetch(self):
        logger.info('Fetching Bugzilla assignment history ...')

        if not dir_exists('./src/its/bugzilla/msr2013'):
            logger.info("Lamfkani's 2013 data set not found. Installing...")
            self._install_msr()

        for project in self._projects:
            self._fetch_assigned_to('./src/its/bugzilla/msr2013/'
                                    + self._get_path() + '/' + project + '/assigned_to.xml')

        for project in self._projects:
            self._fetch_resolutions('./src/its/bugzilla/msr2013/'
                                    + self._get_path() + '/' + project + '/resolution.xml')

    def _install_msr(self):
        os.mkdir('./src/its/bugzilla/msr2013')

        logger.info("Downloading data")
        if not file_exists('./src/its/bugzilla/msr2013/eclipse.tar.gz'):
            download_file(ECLIPSE_URL, './src/its/bugzilla/msr2013/eclipse.tar.gz')
        if not file_exists('./src/its/bugzilla/msr2013/mozilla.tar.gz'):
            download_file(MOZILLA_URL, './src/its/bugzilla/msr2013/mozilla.tar.gz')

        logger.info("Extracting data")

        tar = tarfile.open('./src/its/bugzilla/msr2013/eclipse.tar.gz')
        tar.extractall('./src/its/bugzilla/msr2013')
        tar.close()

        tar = tarfile.open('./src/its/bugzilla/msr2013/mozilla.tar.gz')
        tar.extractall('./src/its/bugzilla/msr2013')
        tar.close()

    def _get_path(self):
        if 'eclipse' in self._domain:
            return 'eclipse'
        else:
            return 'mozilla'

    def _fetch_assigned_to(self, path):
        start_time = time.time()
        logger.info('Importing ' + path + ' ...')

        tree = ElementTree.parse(path)
        root = tree.getroot()

        for node in root.findall('report'):
            bug_id = node.attrib.get('id')

            updates = get_update_events(node)

            self._on_item({
                'bug_id': bug_id,
                'events': updates
            })

        logger.info(
            'Importing ' + path + ' took {:.1f} seconds'.format(time.time() - start_time))

    def _fetch_resolutions(self, path):
        start_time = time.time()
        logger.info('Importing ' + path + ' ...')

        tree = ElementTree.parse(path)
        root = tree.getroot()

        for node in root.findall('report'):
            bug_id = node.attrib.get('id')

            updates = get_update_events(node, resolution_naming)

            self._on_resolution({
                'bug_id': bug_id,
                'events': updates
            })

        logger.info(
            'Importing ' + path + ' took {:.1f} seconds'.format(time.time() - start_time))
