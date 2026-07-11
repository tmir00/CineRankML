FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Full workspace tree is required for `uv sync --frozen` (root pyproject lists all members).
COPY pyproject.toml uv.lock .python-version ./
COPY packages packages
COPY workers workers
COPY jobs jobs
COPY services services
COPY apps apps

RUN uv sync --frozen --package common --no-dev

CMD ["uv", "run", "alembic", "-c", "packages/common/db/alembic.ini", "upgrade", "head"]
