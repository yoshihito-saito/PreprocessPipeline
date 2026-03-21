import importlib.metadata
from packaging.version import InvalidVersion, Version

try:
    __version__ = importlib.metadata.version('kilosort')
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "4.1.2"

try:
    if Version(__version__) < Version("4.0.34"):
        __version__ = "4.1.2"
except InvalidVersion:
    __version__ = "4.1.2"

from .utils import PROBE_DIR, DOWNLOADS_DIR
from .run_kilosort import run_kilosort
from .parameters import DEFAULT_SETTINGS
