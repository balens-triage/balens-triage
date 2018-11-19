import logging
import os
import importlib
from glob import glob
from .load import load_data

logger = logging.getLogger(__name__)


def get_filename(filepath):
    return os.path.basename(filepath).replace('.py', '')


def get_path_only(filepath):
    filename = os.path.basename(filepath)
    return filepath.replace(filename, '')


def get_trainer(classname):
    for file_path in glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "**/*.py")):
        if get_filename(file_path) == classname:
            return load_trainer(file_path, classname)
    else:
        raise Exception('Trainer named ' + classname + ' not found in path ' + './src/train')


path = __file__.replace('__init__.py', '')


def load_trainer(file_path, classname):
    rel_path = file_path.replace(path, '.').replace('/', '.').replace('.py', '')
    module = importlib.import_module(rel_path, package="src.train")

    return getattr(module, classname)
