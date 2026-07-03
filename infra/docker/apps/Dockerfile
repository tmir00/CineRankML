FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ARG PACKAGE
ENV PACKAGE=${PACKAGE}

COPY pyproject.toml uv.lock .python-version ./
COPY packages/common packages/common
COPY apps apps
COPY jobs/train_hybrid_ranker jobs/train_hybrid_ranker

RUN uv sync --frozen --package ${PACKAGE} --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "exec uv run \"$PACKAGE\""]
