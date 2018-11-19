from .hg import MercurialFetcher
from .git import GitFetcher

"""
The purpose of this class is to delegate the extraction of vcs data
"""


class VCSFetcher:
    def __init__(self, config, storage):
        self.config = config

        vcs = config['vcs']
        if vcs['type'] == 'mercurial':
            self._impl = MercurialFetcher(vcs['url'], storage)
        elif vcs['type'] == 'git':
            self._impl = GitFetcher(vcs['url'], storage, config.get_namespace())
        else:
            raise Exception('VCS missing in Config')

    def fetch(self):
        self._impl.fetch()
