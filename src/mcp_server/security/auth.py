# Implements Phase 1, Task 1.4 (Security layer)
# See: /docs/spec.md §6 (Security)
# See: /docs/implementation-plan.md → Task 1.4 DoD & Tests
# JWT authentication middleware

from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.shared.config import Settings, get_settings
from src.shared.observability import get_logger

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)


class JWTAuth:
    """JWT authentication handler"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.secret = self.settings.jwt_secret
        self.algorithm = self.settings.jwt_algorithm

    def create_token(
        self, subject: str, expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT token.

        Args:
            subject: Token subject (typically user ID or client ID)
            expires_delta: Token expiration time delta

        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=60)

        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
        }

        encoded_jwt = jwt.encode(to_encode, self.secret, algorithm=self.algorithm)
        logger.info("JWT token created", subject=subject, expires_at=expire.isoformat())
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            logger.debug("JWT token verified", subject=payload.get("sub"))
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except (jwt.exceptions.DecodeError, jwt.exceptions.InvalidTokenError) as e:
            logger.warning("JWT verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Global auth instance
_jwt_auth: Optional[JWTAuth] = None


def get_jwt_auth() -> JWTAuth:
    """Get global JWT auth instance"""
    global _jwt_auth
    if _jwt_auth is None:
        _jwt_auth = JWTAuth()
    return _jwt_auth


async def verify_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> dict:
    """
    Dependency for JWT token verification.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jwt_auth = get_jwt_auth()
    return jwt_auth.verify_token(credentials.credentials)


async def optional_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> Optional[dict]:
    """
    Optional JWT verification (doesn't raise if missing).

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Decoded token payload or None if not provided
    """
    if credentials is None:
        return None

    try:
        jwt_auth = get_jwt_auth()
        return jwt_auth.verify_token(credentials.credentials)
    except HTTPException:
        return None
