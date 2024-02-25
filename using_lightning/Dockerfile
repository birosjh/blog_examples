FROM python:3.9.13-slim

RUN pip install poetry

WORKDIR /code

COPY pyproject.toml pyproject.toml

RUN poetry install --no-root