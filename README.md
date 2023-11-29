# Blog Examples

This is a repository of code examples from the [Perception-ML](perception-ml.com) website.

Each example is in its own directory, but they have a shared docker environment.

## Environment Setup

Make sure docker is installed before running this.

To start the docker container:

```
docker-compose up -d
```

To ssh into the project:

```
docker-compose exec app bash
```

To shut down the docker container:

```
docker-compose down
```