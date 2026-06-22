FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY packages/common packages/common

RUN uv sync --frozen --package common --no-dev

CMD ["uv", "run", "alembic", "-c", "packages/common/db/alembic.ini", "upgrade", "head"]
