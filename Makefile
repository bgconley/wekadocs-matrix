PHASE?=1
up:
	docker compose up -d
down:
	docker compose down -v
test-phase-%: up
	bash scripts/test/run_phase.sh $* || (echo "Phase $* failed" && exit 1)

# Phase 6 specific targets
test-phase-6: up
	bash scripts/test/run_phase.sh 6 || (echo "Phase 6 failed" && exit 1)
