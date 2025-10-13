# Implements Phase 1, Task 1.4 (Security layer)
# See: /docs/spec.md §6 (Security)
# See: /docs/implementation-plan.md → Task 1.4 DoD & Tests
# Redis token bucket rate limiter

import time
from typing import Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status

from src.shared.config import Config, get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Redis-backed token bucket rate limiter"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.enabled = self.config.rate_limit.enabled
        self.requests_per_minute = self.config.rate_limit.requests_per_minute
        self.burst_size = self.config.rate_limit.burst_size
        self.window_seconds = self.config.rate_limit.window_seconds
        self._redis_client: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis client instance"""
        if self._redis_client is None:
            manager = get_connection_manager()
            self._redis_client = await manager.get_redis_client()
        return self._redis_client

    def _get_client_key(self, request: Request) -> str:
        """
        Extract client identifier from request.
        Uses X-Forwarded-For if available, otherwise client IP.

        Args:
            request: FastAPI request

        Returns:
            Client identifier string
        """
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in the chain
            client_id = forwarded_for.split(",")[0].strip()
        else:
            # Use direct client IP
            client_id = request.client.host if request.client else "unknown"

        # Add authenticated user if available (from JWT)
        user_id = request.state.__dict__.get("user_id")
        if user_id:
            client_id = f"user:{user_id}"

        return client_id

    async def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request is within rate limit using token bucket algorithm.

        Args:
            request: FastAPI request

        Returns:
            True if request is allowed, False if rate limited

        Raises:
            HTTPException: If rate limit exceeded
        """
        if not self.enabled:
            return True

        client_id = self._get_client_key(request)
        redis_key = f"rate_limit:{client_id}"

        try:
            redis = await self._get_redis()
            current_time = time.time()

            # Token bucket algorithm with Redis
            # Key format: rate_limit:<client_id>
            # Value: <tokens>:<last_refill_time>

            # Get current bucket state
            bucket_data = await redis.get(redis_key)

            if bucket_data is None:
                # Initialize new bucket
                tokens = self.burst_size - 1
                last_refill = current_time
                await redis.setex(
                    redis_key, self.window_seconds, f"{tokens}:{last_refill}"
                )
                logger.debug(
                    "Rate limit bucket initialized", client=client_id, tokens=tokens
                )
                return True

            # Parse bucket state
            tokens_str, last_refill_str = bucket_data.split(":")
            tokens = float(tokens_str)
            last_refill = float(last_refill_str)

            # Calculate token refill
            time_passed = current_time - last_refill
            refill_rate = self.requests_per_minute / 60.0  # tokens per second
            tokens_to_add = time_passed * refill_rate
            tokens = min(self.burst_size, tokens + tokens_to_add)

            # Check if request can proceed
            if tokens >= 1.0:
                # Consume one token
                tokens -= 1.0
                await redis.setex(
                    redis_key, self.window_seconds, f"{tokens}:{current_time}"
                )
                logger.debug(
                    "Rate limit check passed",
                    client=client_id,
                    tokens_remaining=round(tokens, 2),
                )
                return True
            else:
                # Rate limit exceeded
                logger.warning(
                    "Rate limit exceeded",
                    client=client_id,
                    tokens=round(tokens, 2),
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={
                        "X-RateLimit-Limit": str(self.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": str(int((1.0 - tokens) / refill_rate)),
                    },
                )

        except HTTPException:
            raise
        except Exception as e:
            # On error, allow request (fail open for availability)
            logger.error("Rate limiter error, allowing request", error=str(e))
            return True


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def check_rate_limit(request: Request) -> bool:
    """
    Dependency for rate limit checking.

    Args:
        request: FastAPI request

    Returns:
        True if allowed

    Raises:
        HTTPException: If rate limit exceeded
    """
    limiter = get_rate_limiter()
    return await limiter.check_rate_limit(request)
