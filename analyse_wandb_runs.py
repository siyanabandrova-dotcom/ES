"""
Analyse W&B runs for es_lora experiments.

For each logged metric (mean_fitness + all eval/* metrics) and for every step,
this script finds the 3 runs that achieved the highest value.

Usage:
    python analyse_wandb_runs.py \
        --project hyperscalees-vllm \
        [--entity my-wandb-entity] \
        [--name-filter some_prefix] \
        [--metrics mean_fitness eval/mean_fitness eval/gsm8k_mean_fitness] \
        [--output results.csv]

If --metrics is not provided, the script auto-discovers all available metrics.
"""

import argparse
import pandas as pd
import wandb
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_runs(project: str, entity: str | None, name_filter: str | None) -> list:
    """Return a list of wandb Run objects matching the given filters."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {}
    if name_filter:
        filters["display_name"] = {"$regex": name_filter}
    runs = api.runs(path, filters=filters if filters else None)
    runs = list(runs)
    print(f"Fetched {len(runs)} runs from '{path}'.")
    return runs


def runs_to_history(runs: list, keys: list[str]) -> dict[str, pd.DataFrame]:
    """
    For each run, download its full history for the requested keys.
    Returns {run_name: DataFrame(columns=[_step, key1, key2, ...])}
    """
    histories = {}
    for run in runs:
        try:
            # samples=None → download every logged row (may be slow for long runs)
            df = run.history(keys=keys, x_axis="_step", pandas=True)
            if df.empty:
                print(f"  [skip] '{run.name}' has no history for requested keys.")
                continue
            df["_run_name"] = run.name
            df["_run_id"] = run.id
            histories[run.name] = df
        except Exception as exc:
            print(f"  [warn] Could not fetch history for '{run.name}': {exc}")
    return histories


def autodiscover_metrics(runs: list) -> list[str]:
    """
    Collect every key that looks like a fitness / eval metric
    from the run summary fields of all runs.
    """
    metric_set: set[str] = set()
    for run in runs:
        for key in run.summary.keys():
            if key.startswith("_"):
                continue
            if "fitness" in key or key.startswith("eval/"):
                metric_set.add(key)
    metrics = sorted(metric_set)
    print(f"Auto-discovered {len(metrics)} metric(s): {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def top3_per_step(
    histories: dict[str, pd.DataFrame],
    metric: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    For a single metric, return a DataFrame with one row per step and
    columns [step, rank1_run, rank1_value, rank2_run, rank2_value, rank3_run, rank3_value].
    """
    # Collect (step, run_name, value) triples
    records = []
    for run_name, df in histories.items():
        if metric not in df.columns:
            continue
        sub = df[["_step", metric]].dropna()
        for _, row in sub.iterrows():
            records.append((int(row["_step"]), run_name, float(row[metric])))

    if not records:
        return pd.DataFrame()

    # Group by step, pick top-N
    step_map: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for step, run_name, value in records:
        step_map[step].append((run_name, value))

    rows = []
    for step in sorted(step_map):
        ranked = sorted(step_map[step], key=lambda x: x[1], reverse=True)[:top_n]
        row: dict = {"step": step}
        for rank_idx, (rname, rval) in enumerate(ranked, start=1):
            row[f"rank{rank_idx}_run"] = rname
            row[f"rank{rank_idx}_value"] = rval
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hyperscalees-vllm",
                        help="W&B project name (default: hyperscalees-vllm)")
    parser.add_argument("--entity", default=None,
                        help="W&B entity / team name (optional)")
    parser.add_argument("--name-filter", default=None,
                        help="Regex to filter run names (optional)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Metrics to analyse. Auto-discovered if not provided.")
    parser.add_argument("--top-n", type=int, default=3,
                        help="How many top runs to report per step (default: 3)")
    parser.add_argument("--output", default="top_runs_per_metric.csv",
                        help="Output CSV path (default: top_runs_per_metric.csv)")
    args = parser.parse_args()

    print(f"Fetching runs from project {args.project} and entity {args.entity} with filter {args.name_filter}")

    # 1. Fetch runs
    runs = fetch_runs(args.project, args.entity, args.name_filter)
    if not runs:
        print("No runs found. Check --project / --entity / --name-filter.")
        return

    # 2. Determine metrics
    metrics = args.metrics if args.metrics else autodiscover_metrics(runs)
    if not metrics:
        print("No fitness / eval metrics found in any run summary.")
        return

    # 3. Download histories (only the columns we need)
    histories = runs_to_history(runs, keys=metrics)
    if not histories:
        print("Could not retrieve history from any run.")
        return

    # 4. Compute top-N per metric per step and accumulate results
    all_rows = []
    for metric in metrics:
        print(f"Analysing metric: {metric} …")
        result_df = top3_per_step(histories, metric, top_n=args.top_n)
        if result_df.empty:
            print(f"  [skip] No data found for metric '{metric}'.")
            continue
        result_df.insert(0, "metric", metric)
        all_rows.append(result_df)
        # Pretty-print a summary: best run at the final step
        last_row = result_df.iloc[-1]
        print(
            f"  Last step ({int(last_row['step'])}): "
            f"#{1} {last_row['rank1_run']} ({last_row['rank1_value']:.4f})"
        )

    if not all_rows:
        print("No results to save.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved results to '{args.output}'  ({len(combined)} rows).")

    # 5. Also print a concise console summary
    print("\n" + "=" * 70)
    print("SUMMARY — Top-3 runs at the LAST available step, per metric")
    print("=" * 70)
    for metric in combined["metric"].unique():
        sub = combined[combined["metric"] == metric]
        last = sub.sort_values("step").iloc[-1]
        print(f"\n  Metric : {metric}")
        print(f"  Step   : {int(last['step'])}")
        for rank in range(1, args.top_n + 1):
            run_col = f"rank{rank}_run"
            val_col = f"rank{rank}_value"
            if run_col in last and pd.notna(last[run_col]):
                print(f"    #{rank}: {last[run_col]}  (value={last[val_col]:.4f})")


if __name__ == "__main__":
    main()