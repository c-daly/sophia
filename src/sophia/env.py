"""Environment helpers for Sophia repository configuration.

This module provides utilities for resolving repository paths and loading
environment configuration in a consistent way across tests and scripts.

The primary use case is the ``SOPHIA_REPO_ROOT`` environment variable which
allows tests to run correctly when the repository is relocated or when
running in CI environments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import cache
from pathlib import Path


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
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def get_env_value(
    key: str,
    env: Mapping[str, str] | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve an env var by checking OS env, stack env, then default.

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
    if key in os.environ:
        return os.environ[key]
    if env and key in env:
        return env[key]
    return default


def get_repo_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the Sophia repo root, honoring SOPHIA_REPO_ROOT if set.

    Priority:
    1. SOPHIA_REPO_ROOT from OS env or provided mapping (if path exists).
    2. GITHUB_WORKSPACE (set by GitHub Actions in CI).
    3. Fallback to parent of this package (works when running from source).

    Args:
        env: Optional mapping to check for SOPHIA_REPO_ROOT.

    Returns:
        Path to the repository root directory.
    """
    env_value = get_env_value("SOPHIA_REPO_ROOT", env)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate

    # GitHub Actions sets GITHUB_WORKSPACE to the repo checkout
    github_workspace = os.getenv("GITHUB_WORKSPACE")
    if github_workspace:
        candidate = Path(github_workspace).resolve()
        if candidate.exists():
            return candidate

    # Fallback: src/sophia/env.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def get_neo4j_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Neo4j connection configuration.

    Loads from environment variables with sensible defaults for testing.

    Args:
        env: Optional mapping from load_stack_env().

    Returns:
        Dictionary with 'uri', 'user', and 'password' keys.
    """
    if env is None:
        env = load_stack_env()

    return {
        "uri": get_env_value("NEO4J_URI", env, "bolt://localhost:7687") or "",
        "user": get_env_value("NEO4J_USER", env, "neo4j") or "",
        "password": get_env_value("NEO4J_PASSWORD", env, "neo4jtest") or "",
    }


def get_milvus_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Milvus connection configuration.

    Loads from environment variables with sensible defaults for testing.

    Args:
        env: Optional mapping from load_stack_env().

    Returns:
        Dictionary with 'host', 'port', and 'healthcheck' keys.
    """
    if env is None:
        env = load_stack_env()

    return {
        "host": get_env_value("MILVUS_HOST", env, "localhost") or "",
        "port": get_env_value("MILVUS_PORT", env, "19530") or "",
        "healthcheck": get_env_value(
            "MILVUS_HEALTHCHECK", env, "http://localhost:9091/healthz"
        )
        or "",
    }
