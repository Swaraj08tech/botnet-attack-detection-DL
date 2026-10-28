"""Password hashing (bcrypt), JWT access tokens, role checks and a simple login rate limiter."""
from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict, deque

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db import AuditLog, User

oauth2 = OAuth2PasswordBearer(tokenUrl='/api/auth/login')


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(user: User) -> str:
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({'sub': str(user.id), 'role': user.role, 'exp': expires}, settings.jwt_secret, algorithm='HS256')


def decode_token(token: str) -> int:
    try:
        return int(jwt.decode(token, settings.jwt_secret, algorithms=['HS256'])['sub'])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Invalid or expired token', headers={'WWW-Authenticate': 'Bearer'}) from None


def get_db(request: Request):
    db = request.app.state.sessions()
    try:
        yield db
    finally:
        db.close()


def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    user = db.get(User, decode_token(token))
    if user is None or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'User not found or inactive')
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != 'admin':
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'Administrator role required')
    return user


def audit(db: Session, user: User | None, action: str, detail: str = ''):
    db.add(AuditLog(user_id=user.id if user else None, action=action, detail=detail[:2000]))
    db.commit()


class LoginLimiter:
    """At most `limit` failed logins per client address in `window` seconds."""

    def __init__(self, limit: int = 10, window: int = 300):
        self.limit, self.window = limit, window
        self.failures: dict[str, deque] = defaultdict(deque)

    def check(self, key: str):
        q = self.failures[key]
        while q and q[0] < time.time() - self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, 'Too many failed logins; try again later')

    def fail(self, key: str):
        self.failures[key].append(time.time())


login_limiter = LoginLimiter()
