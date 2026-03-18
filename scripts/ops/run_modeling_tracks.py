from __future__ import annotations

import argparse

from dengue.forecasting.modeling_tracks import run_modeling_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hierarchy-fixed operational and experimental forecasting tracks."
    )
    parser.add_argument(
        "--config",
        default="configs/modeling_tracks.yaml",
        help="Path to the modeling-track config YAML.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    result = run_modeling_tracks(args.config)
    print(f"[run] run_id      -> {result['run_id']}")
    print(f"[run] run_dir     -> {result['run_dir']}")
    print(f"[run] summary     -> {result['summary_path']}")
    print(f"[run] leaderboard -> {result['leaderboard_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
