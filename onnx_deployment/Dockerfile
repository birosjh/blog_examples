FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3.10 python3-pip python3-opencv

RUN pip install poetry==2.1.3

WORKDIR /code

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root