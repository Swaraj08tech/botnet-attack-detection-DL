"""Settings from environment variables (with safe local-development defaults)."""
from __future__ import annotations

import os
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path):
    """Read KEY=VALUE lines from backend/.env (local development only); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(REPO_ROOT / 'backend/.env')


def _repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


@dataclass
class Settings:
    bundle_dir: Path = field(default_factory=lambda: _repo_path(_env('BUNDLE_DIR', 'serving/bundle')))
    database_url: str = field(default_factory=lambda: _env('DATABASE_URL', f"sqlite:///{(REPO_ROOT / 'backend/local.db').as_posix()}"))
    jwt_secret: str = field(default_factory=lambda: _env('JWT_SECRET', ''))
    jwt_expire_minutes: int = field(default_factory=lambda: int(_env('JWT_EXPIRE_MINUTES', 480)))
    admin_email: str = field(default_factory=lambda: _env('ADMIN_EMAIL', 'admin@example.com'))
    admin_password: str = field(default_factory=lambda: _env('ADMIN_PASSWORD', ''))
    ollama_url: str = field(default_factory=lambda: _env('OLLAMA_URL', 'http://127.0.0.1:11434'))
    llm_model: str = field(default_factory=lambda: _env('LLM_MODEL', 'qwen2.5:7b'))
    cors_origins: list[str] = field(default_factory=lambda: _env('CORS_ORIGINS', 'http://localhost:5173,http://localhost:8080').split(','))
    max_upload_mb: int = field(default_factory=lambda: int(_env('MAX_UPLOAD_MB', 50)))
    max_upload_rows: int = field(default_factory=lambda: int(_env('MAX_UPLOAD_ROWS', 200_000)))
    environment: str = field(default_factory=lambda: _env('ENVIRONMENT', 'development'))
    data_dir: Path = field(default_factory=lambda: _repo_path(_env('DATA_DIR', 'backend/data')))       # staged uploads + PDF reports

    def __post_init__(self):
        self.bundle_dir = _repo_path(self.bundle_dir)
        self.data_dir = _repo_path(self.data_dir)
        if not self.jwt_secret:
            if self.environment == 'production':
                raise RuntimeError('JWT_SECRET must be set in production')
            self.jwt_secret = secrets.token_urlsafe(48)            # development only: tokens reset when the server restarts
            warnings.warn('JWT_SECRET not set: using a random development secret', stacklevel=2)


settings = Settings()
