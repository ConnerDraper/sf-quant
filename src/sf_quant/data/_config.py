import os
from collections.abc import Callable

import dotenv

dotenv.load_dotenv(override=True)

_config = {
    "root": os.environ.get("ROOT", ""),
    "database": os.environ.get("DATABASE", ""),
}

# Cached loaders register their cache_clear here. Pointing the package at a
# different database must drop frames read from the previous one.
_cache_clearers: list[Callable[[], None]] = []


def register_cache(clear: Callable[[], None]) -> None:
    """Register a cache to be dropped whenever the environment changes."""
    _cache_clearers.append(clear)


def clear_cache() -> None:
    """Drop every cached table read. Call after the data on disk changes."""
    for clear in _cache_clearers:
        clear()


def env(root: str | None = None, database: str | None = None) -> None:
    if root is not None:
        _config["root"] = root
    if database is not None:
        _config["database"] = database
    clear_cache()  # cached frames may come from the previous database


class EnvNotConfiguredError(Exception):
    pass


def get_base_path(table_name: str) -> str:
    missing = [key for key in ("root", "database") if not _config[key]]
    if missing:
        missing_vars = ", ".join(var.upper() for var in missing)
        raise EnvNotConfiguredError(
            f"Environment not configured: {missing_vars} not set.\n"
            f"Configure with:\n"
            f"  import sf_quant.data as sfd\n"
            f"  sfd.env(root='/path/to/root', database='your_database')\n"
            f"Or set ROOT and DATABASE environment variables in a .env file."
        )
    return f"{_config['root']}/groups/grp_quant/database/{_config['database']}/{table_name}"
