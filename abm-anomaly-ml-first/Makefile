.PHONY: help build up down logs clean test components

help:
	@echo "ML-First ABM Anomaly Detection System"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean up volumes and images"
	@echo "  make test       - Run tests"
	@echo "  make components - Show status of required components"

build:
	docker compose build

up:
	docker compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "Dashboard: http://localhost:3000"
	@echo "API: http://localhost:8000/docs"
	@echo "Jupyter: http://localhost:8888 (token: ml_jupyter_token_123)"
	@echo "Grafana: http://localhost:3001 (admin/ml_admin)"

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	docker system prune -f

test:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/api/v1/health || echo "API not ready"
	@docker exec abm-ml-postgres pg_isready || echo "Database not ready"

components:
	@echo "Checking required ML components..."
	@echo -n "ml_analyzer.py: "
	@if [ -f "services/anomaly-detector/ml_analyzer.py" ] && [ -s "services/anomaly-detector/ml_analyzer.py" ]; then echo "✓ Present"; else echo "✗ Missing - Add ML analyzer code"; fi
	@echo -n "API endpoints: "
	@if grep -q "expert/anomalies" services/api/main.py 2>/dev/null; then echo "✓ Present"; else echo "✗ Missing - Add expert labeling endpoints"; fi
	@echo -n "Dashboard components: "
	@if grep -q "ExpertLabelingInterface" services/dashboard/src/Dashboard.js 2>/dev/null; then echo "✓ Present"; else echo "✗ Missing - Add dashboard components"; fi

ENV ?= development

flyway-start:
	docker-compose -f docker-compose.yml -f docker-compose-flyway.yml up -d

flyway-stop:
	docker-compose -f docker-compose.yml -f docker-compose-flyway.yml down

migrate:
	docker-compose -f docker-compose-flyway.yml up flyway

dump-schema:
	docker exec abm-ml-postgres pg_dump -U $(POSTGRES_USER) -s $(POSTGRES_DB) > data/db/schema/updated_schema.sql

seed:
	docker-compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f /flyway/sql/R__seed_anomaly_patterns.sql

flyway-logs:
	docker-compose logs -f

rebuild-db:
	docker-compose -f docker-compose-flyway.yml run --rm flyway bash -c \
	"flyway -url=jdbc:postgresql://postgres:5432/$(POSTGRES_DB) \
	        -user=$(POSTGRES_USER) \
	        -password=$(POSTGRES_PASSWORD) \
	        -baselineOnMigrate=true clean migrate"