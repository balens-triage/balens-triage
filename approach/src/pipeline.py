import logging
import re

from src.config import CacheLevel, CacheBackend
from src.storage.caching.get_cache_impl import get_cache_impl
from .storage import LoadStorage
from .train import get_trainer

logger = logging.getLogger(__name__)


def get_step_data(step):
    if isinstance(step, dict):
        return step['name'], step['params']
    else:
        return parse_step(step)


# extract parameter payload of this form from str: trainerClass_paramA,10,.5_
def parse_step(step_str):
    matches = re.findall(r'(.*)_(.*)_', step_str)
    if len(matches) == 0:
        return step_str, []
    else:
        step, params = matches[0]

        return step, parse_params(params)


def parse_params(params):
    params = params.split('+')

    return [parse_value(value) for value in params]


def parse_value(val):
    result = val  # string
    try:
        result = float(val)
    except ValueError:
        return result

    if result.is_integer():
        return int(result)

    return result


class Pipeline:
    def __init__(self, config, project):
        self._config = config
        self._namespace = project.get_namespace()
        self._project = project

    def run(self):
        cache_level = self._config.cache_level or CacheLevel.MEDIUM
        cache_backend = self._config.cache_backend or CacheBackend.LOCAL

        if not self._project.has_pipeline():
            logger.warning('no pipeline specified. skipping.')
            return

        data = {
            'storage': LoadStorage(self._config, self._namespace, self._project.excluded_emails),
            'reports': []
        }

        data['storage'].cache_level = cache_level
        data['storage'].cache_backend = get_cache_impl(cache_backend, self._config)

        steps = [get_step_data(step) for step in self._project['pipeline']]
        classes = self._load_classes(steps)

        logger.info('running pipeline for namespace ' + self._namespace)

        completed_steps = []
        for step in classes:
            step_class, step_params = step
            step_class.cache_level = cache_level
            step_class.cache_backend = get_cache_impl(cache_backend, self._config)
            requires = step_class.requires()

            data = self.preload_steps(requires, data)

            logger.info('Running %s with params %s' % (step_class.__name__,
                                                       ",".join([str(it) for it in step_params])))
            step = step_class(self._namespace, data, *step_params)

            step.load()
            data = step.train()

            completed_steps.append(step.__class__)
            completed_steps += requires

    def preload_steps(self, requires, data):
        names = [get_step_data(step) for step in requires]
        classes = self._load_classes(names)

        # TODO detect cyclical dependencies
        for required in classes:
            logger.info('Injecting required step: ' + required[0].__class__)
            trainer = required[0](self._namespace, data)
            trainer.load()
            data = trainer.train()

        return data

    def _load_classes(self, steps):
        return map(lambda x: (get_trainer(x[0]), x[1]), steps)
