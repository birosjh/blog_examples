# PyTorch Lightning Demo

### Environment Setup

Note:  Use must have Docker and Docker Compose installed to run this.

1. Build the image

```
docker-compose build
```

2. Start the container

```
docker-compose up -d
```

3. Connect with the container

```
docker-compose exec app bash
```

To shut the container down use this command:

```
docker-compose down
```

### Running the Model

To run this model, you must use poetry to run the training script:

```
poetry run python run.py
```

If you want to change any configurations, try training the config.yaml in the `config` file.

### Troubleshooting

If poetry is not installed properly, try running:

```
poetry install --no-root
```