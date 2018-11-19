import os
import logging
from src.config import CacheLevel

try:
    from keras.engine.saving import load_model
except ModuleNotFoundError as e:
    from keras.models import load_model

logger = logging.getLogger(__name__)


# For caching keras models
def cached_kmodel(cachefile, cache_level=CacheLevel.MEDIUM):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args,
                    **kwargs):  # define a wrapper that will finally call "fn" with all arguments
            classname = args[0].__class__.__name__
            namespace = args[0]._ns
            layer = args[0]._layer or ''

            filename = layer + cachefile

            config_cache_level = args[0].cache_level

            cache_backend = args[0].cache_backend

            path = namespace + '/' + classname + '/' + filename

            if cache_backend.exists(path):
                load_path = cache_backend.load(path)
                logger.info("using cached model from '%s'" % filename)
                model = load_model(load_path)
                model._make_predict_function()
                return model

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            logger.info("saving model to cache '%s'" % filename)
            tmp_path = './' + filename
            res.save(tmp_path)

            with open(tmp_path, 'rb') as f:
                cache_backend.write(path, f.read())

            os.remove(tmp_path)

            return res

        return wrapped

    return decorator  # return this "customized" decorator that
