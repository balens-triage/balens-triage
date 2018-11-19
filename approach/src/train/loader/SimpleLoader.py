import logging

from src.base import Trainer

logger = logging.getLogger(__name__)


class SimpleLoader(Trainer):
    def __init__(self, namespace, container):
        super().__init__(namespace)
        self._ns = namespace
        self._container = container

    def train(self):
        logger.info("SimpleLoader has no training step")
        return self._container

    def load(self):
        storage = self._container['storage']

        data_rows = storage.load_bugs_and_history()

        developer_ids = [row['assignee_email'] for row in data_rows]

        self._container['data'] = []
        self._container['target'] = []
        self._container['target_names'] = list(set(developer_ids))

        for bug in data_rows:
            developer = bug['assignee_email']
            vector = bug['summary']
            vector += bug['title']

            for v in bug['comments']:
                vector += str(v) + ' '

            if 'FIXED' in bug['status']:
                self._container['data'].append(vector)
                self._container['target'].append(developer)
        return self._container
