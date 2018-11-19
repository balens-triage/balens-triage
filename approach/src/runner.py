import logging
import datetime

from src.storage.caching.get_cache_impl import get_cache_impl
from src.tools.model.bug_vector import get_bug_vector_from
from src.vcs import VCSFetcher
from .stackoverflow import load_table, load_user_hashes
from .config import load_config, CacheLevel, CacheBackend
from .storage import SaveStorage, LoadStorage, get_connection
from .pipeline import Pipeline
from .its import *

logger = logging.getLogger(__name__)


def overwrite_pipeline(project, pipeline):
    project.pipeline = pipeline
    return project


config = None


class Main:

    def __init__(self, _config, **kwargs):
        self._config = load_config(_config)

        if kwargs.get('pipeline'):
            pipeline = kwargs['pipeline'].split(',')
            self._config.projects = [overwrite_pipeline(project, pipeline) for project in
                                     self._config.projects]

        if kwargs.get('cache_level'):
            self._config.cache_level = CacheLevel[kwargs['cache_level'].upper()]
        else:
            self._config.cache_level = CacheLevel.MEDIUM

        if kwargs.get('maxthreads'):
            self._config.maxThreads = int(kwargs['maxthreads'])

        if kwargs.get('datalimit'):
            self._config.data_limit = int(kwargs['datalimit'])
        else:
            self._config.data_limit = None

        if kwargs.get('cache_backend'):
            self._config.cache_backend = CacheBackend[kwargs['cache_backend'].upper()]
        else:
            self._config.cache_backend = CacheBackend['LOCAL']

        self._loaded_models = {}

    def fetch(self):
        logger.info('Running tasks ...')

        for project in self._config['projects']:
            storage = SaveStorage(self._config['db'], project.get_namespace())

            if project['vcs']:
                VCSFetcher(project, storage).fetch()
            else:
                logger.info('no vcs found, skipping')

            # if project['its']:
            #     ITSFetcher(project, self._config['maxThreads'], storage).fetch()
            # else:
            #     logger.info('no its found, skipping')

    def update(self):
        logger.info('Running project updates ...')

        for project in self._config['projects']:
            storage = SaveStorage(self._config['db'], project.get_namespace())

            if project['its']:
                ITSFetcher(project, self._config['maxThreads'], storage).update()
            else:
                logger.info('no its found, skipping')

    def train(self):
        for project in self._config['projects']:
            pipeline = Pipeline(self._config, project)
            pipeline.run()

    def load_stackoverflow(self):
        conn = get_connection(self._config['db'])
        load_table('Posts', './stackoverflow/Posts.xml', conn)
        load_user_hashes(conn)

    def get_connection(self):
        return get_connection(self._config['db'])

    def train_balens(self, namespace, alpha=0.9):
        storage = LoadStorage(self._config, namespace)

        cache_level = self._config.cache_level
        cache_backend = self._config.cache_backend

        storage.cache_level = cache_level
        storage.cache_backend = get_cache_impl(cache_backend, self._config)

        if namespace not in self._loaded_models:
            self._loaded_models[namespace] = 'LOADING'

            from src.train.ensemble.EnsembleNNBalanced import EnsembleNNBalanced

            model = EnsembleNNBalanced(namespace, {
                'storage': storage
            }, 'lstm', alpha)

            EnsembleNNBalanced.cache_level = cache_level
            EnsembleNNBalanced.cache_backend = get_cache_impl(cache_backend, self._config)

            model.restart()

            self._loaded_models[namespace] = model

        return self._loaded_models[namespace]

    def balens_predict(self, namespace, title, summary):
        if namespace not in self._loaded_models:
            raise Exception('model not yet loaded')
        else:
            bug = {
                'vector': get_bug_vector_from(title, summary),
                'modified_at': datetime.datetime.now(),
                'created_at': datetime.datetime.now(),
                'component': ''
            }

            return self._loaded_models[namespace].ensemble_predict(bug, full_response=True)

    def model_status(self, namespace):
        if namespace not in self._loaded_models:
            return "NOT_LOADED"
        elif self._loaded_models[namespace] == "LOADING":
            return "LOADING"
        else:
            return "LOADED"
