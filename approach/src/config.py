import json

from urllib.parse import urlparse

from src.orderedenum import OrderedEnum


class CacheLevel(OrderedEnum):
    NO_CACHE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class CacheBackend(OrderedEnum):
    LOCAL = 0
    S3 = 1


def load_config(data_input):
    if isinstance(data_input, dict):
        Config(data_input)
    elif isinstance(data_input, str):
        with open(data_input) as f:
            data = json.load(f)
            return Config(data)


class Config:

    def __init__(self, data, **kwargs):
        self.__dict__ = data
        self.__dict__['projects'] = [Project(it) for it in self.__dict__['projects']]

    def __getitem__(self, key):
        return self.__dict__[key]


class Project:

    def __init__(self, data):
        self.__dict__ = data

    def get_namespace(self):
        if self['its']['type'] == 'bugzilla':
            return urlparse(self['its']['url']).netloc.replace(".", "_")
        elif self['its']['type'] == 'github':
            path = urlparse(self['its']['url']).path
            [_, owner, repo] = path.split('/')
            return '_'.join(['github', owner, repo])

    def has_pipeline(self):
        return self['pipeline'] and len(self['pipeline']) > 0

    @property
    def excluded_emails(self):
        if 'ignore' in self.__dict__:
            return self.__dict__['ignore']
        else:
            return []

    def __getitem__(self, key):
        return self.__dict__[key]
