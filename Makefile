.PHONY: deps-base install smoke

PYTORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu126

deps-base:
	pip-compile --strip-extras requirements-base.in -o constraints-base.txt

install:
	@bash -lc "set -euo pipefail; \
		pip install -c constraints-base.txt -r requirements.in --index-url $(PYTORCH_INDEX_URL); \
		pip install --no-deps faiss-gpu-cu12==1.12.0"

smoke:
	bash scripts/00_smoke.sh
