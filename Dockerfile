# Serving image: API, console and analyst. No Java, no Spark.
FROM python:3.12-slim-bookworm AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra agent
COPY src ./src
RUN uv sync --frozen --no-dev --extra agent

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home ope
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/src /app/src
COPY configs /app/configs
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs http://localhost:8000/healthz || exit 1
# models and batch outputs are mounted at /app/artifacts and /app/data (or synced from S3 by an init container)
CMD ["uvicorn", "ope.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
