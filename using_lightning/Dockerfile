FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install python3.9 python3-pip -y

RUN pip3 install poetry

WORKDIR /code

COPY pyproject.toml pyproject.toml

RUN poetry install --no-root