.PHONY: deps-base install smoke

PYTORCH_INDEX_URL_CU128 ?= https://download.pytorch.org/whl/cu128

deps-base:
	pip-compile --strip-extras requirements-base.in -o constraints-base.txt

install:
	@bash -lc "set -euo pipefail; \
		pip install -c constraints-base.txt -r requirements.in --extra-index-url $(PYTORCH_INDEX_URL_CU128); \
		pip install faiss-gpu-cu12 || pip install 'faiss-gpu-cu12[fix_cuda]'"

smoke:
	bash scripts/00_smoke.sh
