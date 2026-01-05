"""Environment helpers for Sophia repository configuration.

This module wraps logos_config to provide Sophia-specific configuration
defaults and convenience functions. It follows the standardization pattern
established in logos #433.

The primary use case is the ``SOPHIA_REPO_ROOT`` environment variable which
allows tests to run correctly when the repository is relocated or when
running in CI environments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from logos_config.env import (
    get_env_value as resolve_env_value,
    get_repo_root as resolve_repo_root,
    load_env_file as resolve_env_file,
)
from logos_config.ports import SOPHIA_PORTS, get_repo_ports

__all__ = [
    "get_env_value",
    "get_repo_root",
    "load_stack_env",
    "get_neo4j_config",
    "get_milvus_config",
    "SOPHIA_PORTS",
    "get_repo_ports",
]


def _default_env_path() -> Path:
    """Get the default path to the stack environment file."""
    override = os.getenv("SOPHIA_STACK_ENV")
    if override:
        return Path(override)
    repo_root = get_repo_root()
    # Standard location for generated stack env file
    candidate = repo_root / "tests" / "e2e" / "stack" / "sophia" / ".env.test"
    if candidate.exists():
        return candidate
    # Fallback to root .env.test (legacy location)
    return repo_root / ".env.test"


@cache
def load_stack_env(env_path: str | Path | None = None) -> dict[str, str]:
    """Load the canonical stack environment (key/value pairs).

    Values are parsed from the ``.env.test`` file. Callers can override the
    location via ``env_path`` or the ``SOPHIA_STACK_ENV`` environment variable.
    Missing files simply yield an empty mapping so tests can still fall back
    to hard-coded defaults.

    Args:
        env_path: Optional path to the environment file.

    Returns:
        Dictionary of environment variable name to value.
    """
    path = Path(env_path) if env_path else _default_env_path()
    if not path.exists():
        return {}
    return cast(dict[str, str], resolve_env_file(path))


def get_env_value(
    key: str,
    env: Mapping[str, str] | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve an env var by checking OS env, stack env, then default.

    Delegates to logos_config.env.get_env_value with type cast.

    Priority order:
    1. OS environment variable
    2. Provided env mapping (e.g., from load_stack_env)
    3. Default value

    Args:
        key: Environment variable name.
        env: Optional mapping to check (typically from load_stack_env).
        default: Default value if not found elsewhere.

    Returns:
        The resolved value or None if not found and no default.
    """
    return cast(str | None, resolve_env_value(key, env, default))


def get_repo_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the Sophia repo root, honoring SOPHIA_REPO_ROOT if set.

    Delegates to logos_config.env.get_repo_root with repo name "sophia".

    Priority:
    1. SOPHIA_REPO_ROOT from OS env or provided mapping (if path exists).
    2. GITHUB_WORKSPACE (set by GitHub Actions in CI).
    3. Fallback to parent of this package (works when running from source).

    Args:
        env: Optional mapping to check for SOPHIA_REPO_ROOT.

    Returns:
        Path to the repository root directory.
    """
    return cast(Path, resolve_repo_root("sophia", env))


def get_neo4j_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Neo4j connection configuration.

    Loads from environment variables with Sophia-specific port defaults.

    Args:
        env: Optional mapping from load_stack_env().

    Returns:
        Dictionary with 'uri', 'user', and 'password' keys.
    """
    ports = get_repo_ports("sophia", env)

    uri = get_env_value(
        "NEO4J_URI",
        env,
        f"bolt://localhost:{ports.neo4j_bolt}",
    )
    user = get_env_value("NEO4J_USER", env, "neo4j")
    password = get_env_value("NEO4J_PASSWORD", env, "neo4jtest")

    assert uri is not None
    assert user is not None
    assert password is not None

    return {"uri": uri, "user": user, "password": password}


def get_milvus_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Milvus connection configuration.

    Loads from environment variables with Sophia-specific port defaults.

    Args:
        env: Optional mapping from load_stack_env().

    Returns:
        Dictionary with 'host', 'port', and 'healthcheck' keys.
    """
    ports = get_repo_ports("sophia", env)

    host = get_env_value("MILVUS_HOST", env, "localhost")
    port = get_env_value("MILVUS_PORT", env, str(ports.milvus_grpc))
    healthcheck = get_env_value(
        "MILVUS_HEALTHCHECK",
        env,
        f"http://localhost:{ports.milvus_metrics}/healthz",
    )

    assert host is not None
    assert port is not None
    assert healthcheck is not None

    return {
        "host": host,
        "port": port,
        "healthcheck": healthcheck,
    }
