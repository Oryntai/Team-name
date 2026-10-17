FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 forte \
    && useradd --uid 10001 --gid forte --create-home forte

WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY backend/ ./backend/
COPY contracts/ ./contracts/
RUN pip install -c requirements.lock . \
    && mkdir -p /app/data /app/frontend/dist \
    && chown -R forte:forte /app/data /home/forte

USER forte
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
