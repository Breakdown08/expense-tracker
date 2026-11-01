FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked

RUN useradd --system --no-create-home tracker
USER tracker
CMD ["/app/.venv/bin/python", "-m", "expense_tracker"]
