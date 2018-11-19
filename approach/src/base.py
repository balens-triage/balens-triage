import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, namespace):
        self._ns = namespace

    @staticmethod
    def requires():
        return []
