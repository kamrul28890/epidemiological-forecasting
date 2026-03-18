from __future__ import annotations

import argparse

from dengue.forecasting.current_forecast import run_current_operational_forecast


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the current 1-4 week operational forecast bundle."
    )
    parser.add_argument(
        "--config",
        default="configs/modeling_tracks.yaml",
        help="Path to the modeling-track config YAML.",
    )
    parser.add_argument(
        "--backtest-run-dir",
        default=None,
        help="Optional run directory from the latest backtest to rank models and choose the primary model.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    result = run_current_operational_forecast(
        config_path=args.config,
        backtest_run_dir=args.backtest_run_dir,
    )
    print(f"[forecast] forecast_run_id -> {result['forecast_run_id']}")
    print(f"[forecast] primary_model   -> {result['primary_model']}")
    print(f"[forecast] report_csv      -> {result['artifacts']['report_csv']}")
    print(f"[forecast] report_md       -> {result['artifacts']['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
