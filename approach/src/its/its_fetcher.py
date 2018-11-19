from .bugzilla import Bugzilla
from .github import GitHub

"""
The purpose of this class is to delegate the extraction of its information
"""


class ITSFetcher:
    def __init__(self, config, max_threads, storage):
        self._config = config

        # TODO create class by type name
        its = config['its']
        if its['type'] == 'bugzilla':
            self._impl = Bugzilla(its, max_threads, storage)
        elif its['type'] == 'github':
            self._impl = GitHub(its, max_threads, storage)
        else:
            raise Exception('Bugzilla missing in Config')

    def fetch(self):
        self._impl.fetch_all()

    def update(self):
        self._impl.update()