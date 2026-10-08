# Scoring service image (the one ECS runs). Untested here: the build machine has no Docker.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8080
# MLFLOW_TRACKING_URI must point at the registry; SERVE_ALIAS pins one alias (ALB canary),
# otherwise the process serves champion + canary with the in-process router.
CMD ["mlops-ref", "serve", "--port", "8080"]
