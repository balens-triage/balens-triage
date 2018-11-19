from botocore.exceptions import ClientError

import boto3
import socket
import os


from src.storage.caching.cache_storage import CacheStorage


class S3(CacheStorage):
    def __init__(self, config):
        super().__init__(config)
        s3_config = config['s3']
        self._key_id = s3_config['keyId']
        self._bucket = s3_config['bucket']
        self._region = s3_config['region']
        self._key_secret = s3_config['keySecret']

        session = boto3.Session(
            aws_access_key_id=self._key_id,
            aws_secret_access_key=self._key_secret,
            region_name=self._region
        )

        self._s3 = session.resource('s3')
        self._tmp_dl = []

        hostname = socket.gethostname()

        # allow env variable overwrite to share caches
        if 'HOST' in os.environ:
            hostname = os.environ['HOST']

        self._prefix = hostname + config['cache']

    def _get_path(self, path):
        return self._prefix + path

    def exists(self, path):
        bucket = self._s3.Bucket(self._bucket)
        key = self._get_path(path)
        objs = list(bucket.objects.filter(Prefix=key))
        return len(objs) > 0 and objs[0].key == key

    def write(self, path, value):
        s3object = self._s3.Object(self._bucket, self._get_path(path))
        s3object.put(Body=value)

    def load(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._s3.Object(self._bucket, self._get_path(path)).download_file(path)
        self._tmp_dl.append(path)

        return path

    def cleanup(self, path):
        if path in self._tmp_dl and os.path.exists(path):
            os.remove(path)
            self._tmp_dl.remove(path)
