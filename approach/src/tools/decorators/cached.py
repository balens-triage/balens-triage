import pickle

from src.config import CacheLevel


def cached(cachefile, cache_level=CacheLevel.MEDIUM):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args,
                    **kwargs):  # define a wrapper that will finally call "fn" with all arguments
            classname = args[0].__class__.__name__
            namespace = args[0]._ns
            config_cache_level = args[0].cache_level

            cache_backend = args[0].cache_backend

            path = namespace + '/' + classname + '/' + cachefile

            if cache_backend.exists(path):
                path = cache_backend.load(path)
                with open(path, 'rb') as cachehandle:
                    result = pickle.load(cachehandle, encoding="bytes")
                    cache_backend.cleanup(path)
                    return result

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            cache_backend.write(path, pickle.dumps(res, protocol=4))

            return res

        return wrapped

    return decorator  # return this "customized" decorator that


def mem_cached():
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        cache = {}

        def wrapped(*args):  # define a wrapper that will finally call "fn" with all arguments
            key = hash(args)

            if key in cache:
                return cache[key]
            else:
                res = fn(*args)
                cache[key] = res
                return res

        return wrapped

    return decorator  # return this "customized" decorator that
