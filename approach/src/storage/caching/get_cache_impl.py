from src.config import CacheBackend
from src.storage.caching.local import Local
from src.storage.caching.s3 import S3

cache_switch = {
    str(CacheBackend.LOCAL): Local,
    str(CacheBackend.S3): S3
}


# This is the deep learning text classification layer
def get_cache_impl(_type, config):
    return cache_switch.get(str(_type))(config)
