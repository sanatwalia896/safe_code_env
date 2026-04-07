ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=safe_code_env

COPY . /app/env
WORKDIR /app/env

# ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi

# ── BAKE BGE MODEL INTO IMAGE ─────────────────────────────────
# This runs during build so the model is IN the image
# No download needed at runtime — critical for 20min limit
RUN /app/env/.venv/bin/python -c "from fastembed import TextEmbedding; print('Downloading BGE model...'); model = TextEmbedding('BAAI/bge-small-en-v1.5'); list(model.embed(['test'])); print('BGE model baked into image successfully')"
RUN mkdir -p /root/.cache/fastembed

# ── Final runtime stage ───────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# copy venv with BGE model already inside
COPY --from=builder /app/env/.venv /app/.venv

# copy environment code
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# BGE model cache location — fastembed stores here by default
# copy from builder so it's available at runtime
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /root/.cache/fastembed   /root/.cache/fastembed

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
