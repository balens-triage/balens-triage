from src.storage.caching.cache_storage import CacheStorage
import os
import logging

logger = logging.getLogger(__name__)


class Local(CacheStorage):
    def __init__(self, config):
        super().__init__(config)
        self._prefix = config['cache']

    def write(self, path, value):
        path = self._get_path(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as file:
            logger.info("saving result to local cache '%s'" % path)
            file.write(value)

    def _get_path(self, path):
        return self._prefix + path

    def exists(self, path):
        path = self._get_path(path)
        return os.path.exists(path)

    def load(self, path):
        path = self._get_path(path)
        logger.info("using locally cached result from '%s'" % path)
        return path

    def cleanup(self, path):
        pass
