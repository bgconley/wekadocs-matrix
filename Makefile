PHASE?=1
up:
	docker compose up -d
down:
	docker compose down -v
test-phase-%: up
	bash scripts/test/run_phase.sh $* || (echo "Phase $* failed" && exit 1)
