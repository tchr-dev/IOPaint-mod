"""Session management for budget tracking."""

import hashlib
import uuid
from typing import Optional
from fastapi import Request, Header


def generate_session_id() -> str:
    """Generate a new random session ID.

    Returns:
        UUID4 string suitable for use as session identifier
    """
    return str(uuid.uuid4())


def get_session_id_from_request(
    request: Request,
    x_session_id: Optional[str] = None,
) -> str:
    """Extract session ID from request or generate fallback.

    Session ID precedence:
    1. X-Session-Id header (explicit session)
    2. session_id query parameter
    3. Fallback: hash of client IP + User-Agent (pseudo-session)

    Args:
        request: FastAPI request object
        x_session_id: Optional header value if already extracted

    Returns:
        Session identifier string
    """
    # Try header first (may be passed directly)
    if x_session_id:
        return x_session_id

    # Try from request headers
    session_id = request.headers.get("X-Session-Id")
    if session_id:
        return session_id

    # Try query parameter
    session_id = request.query_params.get("session_id")
    if session_id:
        return session_id

    # Fallback: pseudo-session from client fingerprint
    return _generate_fallback_session_id(request)


def _generate_fallback_session_id(request: Request) -> str:
    """Generate a fallback session ID from request fingerprint.

    Uses client IP and User-Agent to create a consistent pseudo-session
    for clients that don't send explicit session IDs.

    Args:
        request: FastAPI request object

    Returns:
        16-character hex string
    """
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    fingerprint = f"{client_host}:{user_agent}"
    return hashlib.md5(fingerprint.encode()).hexdigest()[:16]


# FastAPI dependency for extracting session ID
async def get_session_id(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> str:
    """FastAPI dependency for extracting session ID.

    Usage in endpoint:
        @app.get("/api/v1/example")
        async def example(session_id: str = Depends(get_session_id)):
            ...
    """
    return get_session_id_from_request(request, x_session_id)
