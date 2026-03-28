"""
Rank W&B runs by their combined performance across all math-eval metrics.

For every eval step the script:
  1. Collects per-metric scores for every run.
  2. Computes a *composite score* for each (run, step) pair by averaging the
     run's value across all metrics present at that step.
  3. Ranks (run, step) pairs by composite score and reports the global top-N.

Additionally, for every eval step a per-step ranking is printed so you can see
which run was best at each point in training.

Metric selection (in priority order):
  1. --metrics a b c    explicit list
  2. --default-metrics  use hardcoded MATH_EVAL_METRICS
  3. (nothing)          auto-discover all eval/*_mean_fitness from run summaries

Usage:
    python rank_eval_runs.py \\
        --project hyperscalees-vllm \\
        [--entity my-team] \\
        [--name-filter "rebuttal-(deepscaler|orz)-math2"] \\
        [--default-metrics] \\
        [--metrics eval/gsm8k_mean_fitness eval/aime24_mean_fitness ...] \\
        [--top-n 5] \\
        [--aggregate mean|sum] \\
        [--output ranked_runs.csv] \\
        [--global-output global_top_runs.csv]
"""

import argparse

import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_METRIC = "total_time"  # Cumulative wall-clock seconds logged at every train step

MATH_EVAL_SPLITS = [
    "math", "amc", "olympiad_bench", "minerva",
    "aime24", "aime25",
]  # gsm8k / asdiv excluded — adjust here if needed
MATH_EVAL_METRICS = [f"eval/{s}_mean_fitness" for s in MATH_EVAL_SPLITS]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_runs(project: str, entity: str | None, name_filter: str | None) -> list:
    """Return wandb Run objects matching the filters."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters: dict = {}
    if name_filter:
        filters["display_name"] = {"$regex": name_filter}
    runs = list(api.runs(path, filters=filters or None))
    print(f"Fetched {len(runs)} run(s) from '{path}'.")
    return runs


def autodiscover_metrics(runs: list) -> list[str]:
    """
    Return benchmark eval metrics found in any run's summary, intersected
    with MATH_EVAL_METRICS so we never pick up unexpected extra metrics.
    """
    metric_set: set[str] = set()
    for run in runs:
        for key in run.summary.keys():
            if key.startswith("eval/") and key.endswith("_mean_fitness"):
                metric_set.add(key)
    metrics = sorted(metric_set & set(MATH_EVAL_METRICS))
    print(f"Auto-discovered {len(metrics)} benchmark metric(s): {metrics}")
    return metrics


def fetch_histories(runs: list, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """Download per-run history for the requested metric columns + total_time."""
    histories: dict[str, pd.DataFrame] = {}
    keys = metrics + [TIME_METRIC]
    for run in runs:
        try:
            df = run.history(keys=keys, x_axis="_step", pandas=True)
            if df.empty:
                print(f"  [skip] '{run.name}' -- no history for requested metrics.")
                continue
            df["_run_name"] = run.name
            df["_run_id"] = run.id
            histories[run.name] = df
        except Exception as exc:
            print(f"  [warn] Could not fetch history for '{run.name}': {exc}")
    print(f"Retrieved history for {len(histories)} run(s).")
    return histories


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def build_scores_table(
    histories: dict[str, pd.DataFrame],
    metrics: list[str],
    aggregate: str = "mean",
) -> pd.DataFrame:
    """
    Build a wide table of (step, run, training_time_s, composite_score, <metric cols>).

    composite_score = mean (or sum) of the run's eval metric values at that step.
    training_time_s = total_time logged at that step (cumulative wall-clock seconds).
    """
    records = []
    for run_name, df in histories.items():
        present_metrics = [m for m in metrics if m in df.columns]
        if not present_metrics:
            continue
        cols = ["_step"] + present_metrics
        if TIME_METRIC in df.columns:
            cols.append(TIME_METRIC)
        sub = df[cols].copy()
        sub["_run_name"] = run_name
        records.append(sub)

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)

    metric_cols = [m for m in metrics if m in combined.columns]
    if aggregate == "mean":
        combined["composite_score"] = combined[metric_cols].mean(axis=1, skipna=True)
    elif aggregate == "sum":
        combined["composite_score"] = combined[metric_cols].sum(axis=1, skipna=True, min_count=1)
    else:
        raise ValueError(f"Unknown aggregate='{aggregate}'. Use 'mean' or 'sum'.")

    combined = combined.rename(columns={
        "_step": "step",
        "_run_name": "run",
        TIME_METRIC: "training_time_s",
    })
    combined = combined.dropna(subset=["composite_score"])
    combined = combined.sort_values(["step", "composite_score"], ascending=[True, False])
    return combined


# ---------------------------------------------------------------------------
# Per-step ranking
# ---------------------------------------------------------------------------

def rank_per_step(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    For each step, keep the top-N runs by composite_score.
    Returns a DataFrame with columns:
        step, rank, run, training_time_s, training_time_h, composite_score, <metric cols...>
    """
    rows = []
    meta_cols = {"step", "run", "composite_score", "training_time_s"}
    metric_cols = [c for c in scores.columns if c not in meta_cols]
    for step, grp in scores.groupby("step"):
        grp_sorted = grp.sort_values("composite_score", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(grp_sorted.iterrows(), start=1):
            t = row.get("training_time_s", float("nan"))
            entry: dict = {
                "step": int(step),
                "rank": rank,
                "run": row["run"],
                "training_time_s": t,
                "training_time_h": t / 3600 if pd.notna(t) else float("nan"),
                "composite_score": row["composite_score"],
            }
            for m in metric_cols:
                entry[m] = row.get(m, float("nan"))
            rows.append(entry)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Global ranking: best (run, step) pairs overall
# ---------------------------------------------------------------------------

def global_top_n(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Find the top-N (run, step) pairs across all steps by composite_score.
    Ties broken by step (earlier step wins -- conservative choice for papers).
    """
    sorted_all = scores.sort_values(
        ["composite_score", "step"],
        ascending=[False, True],
    ).head(top_n)
    sorted_all = sorted_all.reset_index(drop=True)
    sorted_all.index += 1  # 1-based rank
    sorted_all.index.name = "global_rank"
    return sorted_all


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as Xh Ym Zs, or '?' if missing."""
    if not pd.notna(seconds):
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def print_per_step_summary(ranked: pd.DataFrame, top_n: int) -> None:
    meta_cols = {"step", "rank", "run", "composite_score", "training_time_s", "training_time_h"}
    metric_cols = [c for c in ranked.columns if c not in meta_cols]
    for step in sorted(ranked["step"].unique()):
        grp = ranked[ranked["step"] == step].sort_values("rank")
        print(f"\n  Step {step}")
        for _, row in grp.iterrows():
            metric_str = "  ".join(
                f"{m.split('/')[-1]}={row[m]:.4f}"
                for m in metric_cols
                if pd.notna(row.get(m))
            )
            time_str = _fmt_time(row.get("training_time_s", float("nan")))
            print(f"    #{int(row['rank'])}: {row['run']}"
                  f"  composite={row['composite_score']:.4f}"
                  f"  time={time_str}"
                  + (f"  [{metric_str}]" if metric_str else ""))


def print_global_summary(top: pd.DataFrame) -> None:
    meta_cols = {"step", "run", "composite_score", "training_time_s", "training_time_h"}
    metric_cols = [c for c in top.columns if c not in meta_cols]
    print(f"\n{'='*70}")
    print("GLOBAL TOP RUNS  (best composite score across all eval steps)")
    print(f"{'='*70}")
    for rank, row in top.iterrows():
        metric_str = "  ".join(
            f"{m.split('/')[-1]}={row[m]:.4f}"
            for m in metric_cols
            if pd.notna(row.get(m))
        )
        time_str = _fmt_time(row.get("training_time_s", float("nan")))
        print(f"  #{rank}: {row['run']}"
              f"  @step={int(row['step'])}"
              f"  composite={row['composite_score']:.4f}"
              f"  time={time_str}")
        if metric_str:
            print(f"       [{metric_str}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", default="hyperscalees-vllm",
                        help="W&B project name  (default: hyperscalees-vllm)")
    parser.add_argument("--entity", default=None,
                        help="W&B entity / team name  (optional)")
    parser.add_argument("--name-filter", default=None,
                        help="Regex applied to run display names  (optional)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Explicit list of W&B metrics to use. "
                             "Takes priority over --default-metrics and auto-discovery.")
    parser.add_argument("--top-n", type=int, default=3,
                        help="How many top runs to report  (default: 3)")
    parser.add_argument("--aggregate", choices=["mean", "sum"], default="mean",
                        help="How to combine per-metric scores into a composite "
                             "(default: mean)")
    parser.add_argument("--output", default="ranked_eval_runs.csv",
                        help="Path for the per-step ranking CSV  "
                             "(default: ranked_eval_runs.csv)")
    parser.add_argument("--global-output", default="global_top_runs.csv",
                        help="Path for the global top-N CSV  "
                             "(default: global_top_runs.csv)")
    args = parser.parse_args()

    # 1. Fetch runs ----------------------------------------------------------
    print(f"\nProject : {args.project}")
    print(f"Entity  : {args.entity or '(default)'}")
    print(f"Filter  : {args.name_filter or '(none)'}\n")
    runs = fetch_runs(args.project, args.entity, args.name_filter)
    if not runs:
        print("No runs found. Adjust --project / --entity / --name-filter.")
        return

    # 2. Resolve metrics -----------------------------------------------------
    if args.metrics:
        metrics = args.metrics
        print(f"Using {len(metrics)} user-specified metric(s): {metrics}")
    else:
        metrics = autodiscover_metrics(runs)
    if not metrics:
        print("No eval metrics found. "
              "Pass --metrics explicitly or check your W&B runs.")
        return

    # 3. Download histories --------------------------------------------------
    histories = fetch_histories(runs, metrics)
    if not histories:
        print("No history retrieved. Cannot produce rankings.")
        return

    # 4. Build composite score table -----------------------------------------
    scores = build_scores_table(histories, metrics, aggregate=args.aggregate)
    if scores.empty:
        print("Scores table is empty -- no overlapping (step, metric) data found.")
        return

    # 5. Per-step ranking ----------------------------------------------------
    ranked = rank_per_step(scores, top_n=args.top_n)
    ranked.to_csv(args.output, index=False)
    print(f"\nPer-step ranking saved to '{args.output}'  ({len(ranked)} rows).")

    # 6. Global top-N --------------------------------------------------------
    global_top = global_top_n(scores, top_n=args.top_n)
    global_top.to_csv(args.global_output)
    print(f"Global top-{args.top_n} saved to '{args.global_output}'.")

    # 7. Console summaries ---------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PER-STEP TOP-{args.top_n}  (composite = {args.aggregate} of {len(metrics)} metric(s))")
    print(f"{'='*70}")
    print_per_step_summary(ranked, args.top_n)
    print_global_summary(global_top)

    # 8. Paper-friendly one-liner: single best (run, step) -------------------
    best = global_top.iloc[0]
    print(f"\n{'='*70}")
    print("BEST CHECKPOINT FOR PAPER")
    print(f"{'='*70}")
    print(f"  Run            : {best['run']}")
    print(f"  Step           : {int(best['step'])}")
    print(f"  Training time  : {_fmt_time(best.get('training_time_s', float('nan')))} "
          f"({best.get('training_time_s', float('nan')):.0f}s)")
    print(f"  Composite ({args.aggregate}) : {best['composite_score']:.4f}")
    meta_cols = {"step", "run", "composite_score", "training_time_s", "training_time_h"}
    metric_cols = [c for c in global_top.columns if c not in meta_cols]
    for m in metric_cols:
        if pd.notna(best.get(m)):
            print(f"  {m.split('eval/')[-1]:35s}: {best[m]:.4f}")


if __name__ == "__main__":
    main()