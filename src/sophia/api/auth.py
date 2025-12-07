"""Authentication middleware for Sophia API."""

from typing import Callable, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from logos_config import get_env_value


security = HTTPBearer()


def get_api_token() -> str:
    """Get the API token from environment variable.

    Returns:
        API token string

    Raises:
        RuntimeError: If SOPHIA_API_TOKEN is not set
    """
    token = get_env_value("SOPHIA_API_TOKEN")
    if not token:
        raise RuntimeError(
            "SOPHIA_API_TOKEN environment variable must be set for authentication"
        )
    return token


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify the bearer token in the request.

    Args:
        credentials: HTTP authorization credentials from request

    Returns:
        Verified token string

    Raises:
        HTTPException: If token is invalid or missing
    """
    expected_token = get_api_token()
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token",
        )
    return credentials.credentials


def create_auth_dependency(required: bool = True) -> Callable:
    """Create an authentication dependency.

    Args:
        required: Whether authentication is required

    Returns:
        Authentication dependency function
    """
    if required:
        return verify_token

    # Optional authentication - returns None if not provided
    async def optional_verify(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(
            HTTPBearer(auto_error=False)
        ),
    ) -> Optional[str]:
        if credentials is None:
            return None
        return verify_token(credentials)

    return optional_verify
