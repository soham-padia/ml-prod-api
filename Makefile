.PHONY: install dev test lint format run docker-build docker-up docker-down

install:
	pip install -r requirements.txt -r requirements-dev.txt

dev:
	uvicorn ml_prod_api.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

run:
	APP_ENV=local MODEL_PROVIDER=stub uvicorn ml_prod_api.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ml-prod-api:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
