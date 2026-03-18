.PHONY: glm gbm sarima summarize compare stack baselines sweep tracks forecast-current paper-assets

glm:
	@for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_glm --fold $$f; done

gbm:
	@for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_gbm --fold $$f; done

sarima:
	@for f in F1 F2 F3 F4 F5; do python -u -m scripts.training.train_sarima --fold $$f; done

summarize:
	@python -u -m scripts.analysis.summarize_glm

compare:
	@python -u -m scripts.analysis.stack_and_compare --compare-only

stack:
	@python -u -m scripts.analysis.stack_and_compare --stack

baselines: glm gbm compare stack

sweep:
	@bash scripts/ops/run_glm_alpha_sweep.sh

tracks:
	@python -u -m scripts.ops.run_modeling_tracks

forecast-current:
	@python -u -m scripts.forecasting.forecast_operational_current

paper-assets:
	@python -u -m scripts.analysis.generate_paper_assets
