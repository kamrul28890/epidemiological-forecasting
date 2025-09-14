import glob
import os

import pandas as pd


def load_model(model: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(f"artifacts/tables/metrics_{model}_F*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
            continue
        fold = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        df["model"] = model
        df["fold"] = fold
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["model", "fold", "role", "horizon", "MAE", "RMSE"])


glm = load_model("glm")
gbm = load_model("gbm")
both = pd.concat([glm, gbm], ignore_index=True)

for c in ("MAE", "RMSE"):
    if c in both.columns:
        both[c] = pd.to_numeric(both[c], errors="coerce")

val = both[both.get("role", "") == "val"].copy()

overall = (
    val[val["horizon"].isna()]
    .groupby("model")
    .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"))
    .reset_index()
)
foldwise = (
    val[val["horizon"].isna()]
    .groupby(["model", "fold"])
    .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"))
    .reset_index()
)
by_h = (
    val[val["horizon"].notna()]
    .groupby(["model", "horizon"])
    .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"))
    .reset_index()
)

print("\n== Overall VAL leaderboard ==")
print(overall.sort_values("MAE").to_string(index=False))

print("\n== Per-fold (VAL, overall) ==")
print(foldwise.sort_values(["fold", "model"]).to_string(index=False))

print("\n== By horizon (VAL, mean across folds) ==")
print(by_h.sort_values(["horizon", "model"]).to_string(index=False))

os.makedirs("reports/metrics", exist_ok=True)
overall.to_csv("reports/metrics/leaderboard_overall.csv", index=False)
foldwise.to_csv("reports/metrics/leaderboard_by_fold.csv", index=False)
by_h.to_csv("reports/metrics/leaderboard_by_horizon.csv", index=False)
