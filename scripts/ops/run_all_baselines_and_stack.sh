#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$PWD"

for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_glm --fold "$f"; done
for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_gbm --fold "$f"; done

# for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_sarima --fold "$f"; done

python -u -m scripts.analysis.stack_ensemble
python -u -m scripts.analysis.summarize_gbm
python -u -m scripts.analysis.summarize_glm
python -u -m scripts.analysis.summarize_models
