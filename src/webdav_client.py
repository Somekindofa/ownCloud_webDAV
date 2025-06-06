from typing import List
from tqdm import tqdm
from webdav3.client import Client
from .config import WEBDAV_URL, WEBDAV_USERNAME, WEBDAV_PASSWORD, DEFAULT_CHUNK

_options = {
    "webdav_hostname": WEBDAV_URL,
    "webdav_login":    WEBDAV_USERNAME,
    "webdav_password": WEBDAV_PASSWORD,
    "disable_check":   True,          # skip PROPFIND auth probe (ownCloud quirk)
}

_client = Client(_options)


# -- S C R U D -----------------------------------------------------------
def list_dir(path: str = "/") -> List[str]:
    """Return list of files/folders at remote path."""
    return _client.list(path)


def upload(local_path: str, remote_path: str) -> None:
    """Create or replace a file in ownCloud."""
    _client.upload_sync(remote_path=remote_path, local_path=local_path)


def download(remote_path: str, local_path: str) -> None:
    """Fetch remote file to local disk with progress bar."""
    size = _client.info(remote_path)["size"]
    with tqdm(total=int(size), unit="B", unit_scale=True, desc=remote_path) as bar:
        with open(local_path, "wb") as f:
            _client.download_from(
                buff=f,
                remote_path=remote_path,
                progress=lambda current, total: bar.update(current - bar.n),
            )


def delete(remote_path: str) -> None:
    _client.clean(remote_path)


def exists(remote_path: str) -> bool:
    return _client.check(remote_path)
