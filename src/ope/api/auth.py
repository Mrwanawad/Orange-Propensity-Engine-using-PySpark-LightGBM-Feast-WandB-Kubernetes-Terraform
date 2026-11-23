"""OAuth2 client-credentials JWT validation, with a static API key fallback for local use,
plus a fixed-window rate limit per client.

Production: the caller gets a token from the identity provider (client credentials grant) and
sends `Authorization: Bearer <jwt>`. We check signature (JWKS), issuer, audience and expiry;
the client id is the `azp` / `client_id` / `sub` claim.
Local: `X-API-Key: <key>` against OPE_API_KEYS (comma separated).
"""

from __future__ import annotations

import os
import time
from functools import cache

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)
api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
RATE_PER_MINUTE = int(os.environ.get("OPE_RATE_LIMIT_PER_MINUTE", "1200"))


@cache
def _jwks():
    url = os.environ.get("OPE_JWKS_URL")
    return jwt.PyJWKClient(url) if url else None


def _keys() -> set[str]:
    return {k.strip() for k in os.environ.get("OPE_API_KEYS", "").split(",") if k.strip()}


def _verify_jwt(token: str) -> str:
    client = _jwks()
    if client is None:
        raise HTTPException(401, "JWT auth not configured")
    try:
        key = client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256"],
            audience=os.environ.get("OPE_JWT_AUDIENCE"),
            issuer=os.environ.get("OPE_JWT_ISSUER"),
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid token: {e.__class__.__name__}") from None
    return str(claims.get("azp") or claims.get("client_id") or claims["sub"])


class RateLimiter:
    """Fixed one-minute window. Redis when available (shared across replicas), memory otherwise."""

    def __init__(self, limit: int):
        self.limit = limit
        self.mem: dict[tuple[str, int], int] = {}

    def hit(self, client: str, redis=None) -> bool:
        window = int(time.time() // 60)
        if redis is not None:
            try:
                key = f"ope:rl:{client}:{window}"
                pipe = redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, 120)
                count = pipe.execute()[0]
                return count <= self.limit
            except Exception:
                pass
        k = (client, window)
        self.mem[k] = self.mem.get(k, 0) + 1
        if len(self.mem) > 10_000:
            self.mem = {kk: v for kk, v in self.mem.items() if kk[1] == window}
        return self.mem[k] <= self.limit


limiter = RateLimiter(RATE_PER_MINUTE)


def client_id(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    key: str | None = Depends(api_key),
) -> str:
    if creds is not None:
        cid = _verify_jwt(creds.credentials)
    elif key is not None and key in _keys():
        cid = f"key:{key[:4]}"
    else:
        raise HTTPException(401, "missing or invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    if not limiter.hit(cid, getattr(request.app.state, "redis", None)):
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
    request.state.client_id = cid
    return cid
