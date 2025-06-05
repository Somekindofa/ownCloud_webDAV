from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

WEBDAV_URL      = os.getenv("WEBDAV_URL")      # e.g. https://owncloud.mines.fr/remote.php/webdav
WEBDAV_USERNAME = os.getenv("WEBDAV_USERNAME")
WEBDAV_PASSWORD = os.getenv("WEBDAV_PASSWORD")

DEFAULT_CHUNK   = 1024 * 1024  # 1 MiB
