from .run import run as default_run
from .sog_run import run as sog_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["sog"] = sog_run
