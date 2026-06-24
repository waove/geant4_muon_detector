#!/usr/bin/env python3
"""
merge_runs.py — Merge summary CSVs from parallel CosmicMuonSim runs.

Merges the five run-level summary/data files:
  - edep_summary.csv      (summed totals, recomputed mean)
  - stopping.csv          (summed counts, recomputed fractions)
  - fluence.csv           (summed step lengths, recomputed fluence)
  - entry_ke.csv          (concatenated raw records — no aggregation needed)
  - gun_energy.csv        (concatenated raw records — no aggregation needed)

Per-event and step-level files (edep_per_event.csv, steps.csv,
trajectories.csv) are NOT merged — they are capped at a few thousand
events and can be taken from any single run for detailed plotting.

Usage:
    python merge_runs.py run1/results run2/results run3/results
    python merge_runs.py run*/results                    # glob
    python merge_runs.py run*/results -o merged_results  # custom output dir
"""

import argparse
import shutil
import sys
import numpy as np
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load(run_dirs: list[Path], filename: str) -> list[pd.DataFrame]:
    dfs = []
    for d in run_dirs:
        p = d / filename
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print(f"  warning: {p} not found — skipping this run for {filename}")
    return dfs


def count_events(run_dirs: list[Path]) -> tuple[int, list[int]]:
    """
    Infer the number of events per run from edep_summary.csv.
    total_edep / mean_edep = n_events for any nonzero row.
    Falls back to entry_ke.csv event count if edep_summary is unavailable.
    """
    per_run = []
    for d in run_dirs:
        n = 0
        p = d / "edep_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            nonzero = df[df["mean_edep_MeV"] > 0]
            if not nonzero.empty:
                row = nonzero.iloc[0]
                n = round(row["total_edep_MeV"] / row["mean_edep_MeV"])
        # Fallback: count unique events in entry_ke.csv
        if n == 0:
            pk = d / "entry_ke.csv"
            if pk.exists():
                ek = pd.read_csv(pk)
                n = ek["eventID"].nunique()
        per_run.append(int(n))
    return sum(per_run), per_run


def _concat_with_offset(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    """
    Concatenate a raw-record CSV across runs, offsetting eventID per run
    so IDs remain unique in the merged file.
    Handles runs where the file exists but contains no data rows.
    """
    dfs = _load(run_dirs, filename)
    if not dfs:
        return pd.DataFrame()

    offset = 0
    shifted = []
    for df in dfs:
        if df.empty:
            continue
        df = df.copy()
        if "eventID" in df.columns:
            df["eventID"] = df["eventID"] + offset
            max_id = df["eventID"].max()
            offset += int(max_id) + 1 if pd.notna(max_id) else 1
        shifted.append(df)

    if not shifted:
        return pd.DataFrame()
    return pd.concat(shifted, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file merge functions
# ──────────────────────────────────────────────────────────────────────────────

def merge_edep_summary(run_dirs: list[Path], n_events_total: int) -> pd.DataFrame:
    """Sum total_edep_MeV across runs, recompute mean."""
    dfs = _load(run_dirs, "edep_summary.csv")
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0][["volume", "particle"]].copy()
    merged["total_edep_MeV"] = sum(df["total_edep_MeV"] for df in dfs)
    merged["mean_edep_MeV"]  = merged["total_edep_MeV"] / max(n_events_total, 1)
    return merged


def merge_stopping(run_dirs: list[Path], n_events_total: int) -> pd.DataFrame:
    """
    Sum n_entered and n_stopped, recompute both fractions.

    stop_frac           = n_stopped[vol] / n_entered[vol]
    stop_frac_vs_scint0 = n_stopped[vol] / n_entered[Scint_0]  (per particle species)
    """
    dfs = _load(run_dirs, "stopping.csv")
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0][["volume", "particle"]].copy()
    merged["n_entered"] = sum(df["n_entered"] for df in dfs)
    merged["n_stopped"] = sum(df["n_stopped"] for df in dfs)

    merged["stop_frac"] = merged["n_stopped"] / merged["n_entered"].replace(0, 1)

    def _scint0_entered(row):
        ref = merged.loc[
            (merged["volume"] == "Scint_0") &
            (merged["particle"] == row["particle"]),
            "n_entered"
        ]
        return ref.values[0] if len(ref) else 1

    merged["stop_frac_vs_scint0"] = merged.apply(
        lambda row: row["n_stopped"] / max(_scint0_entered(row), 1), axis=1
    )

    return merged


def merge_fluence(run_dirs: list[Path], n_events_total: int) -> pd.DataFrame:
    """Sum total_steplen_mm, recompute fluence."""
    dfs = _load(run_dirs, "fluence.csv")
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0][["volume", "particle", "volume_mm3"]].copy()
    merged["total_steplen_mm"] = sum(df["total_steplen_mm"] for df in dfs)
    vol = merged["volume_mm3"].replace(0, 1)
    merged["fluence_per_mm2"]           = merged["total_steplen_mm"] / vol
    merged["fluence_per_event_per_mm2"] = (
        merged["fluence_per_mm2"] / max(n_events_total, 1)
    )
    return merged


def merge_stopping_conditional(run_dirs: list[Path]) -> pd.DataFrame:
    """
    Sum n_scint0_entered and n_stopped_here across runs, recompute
    cond_stop_frac = n_stopped_here / n_scint0_entered.
    """
    dfs = _load(run_dirs, "stopping_conditional.csv")
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0][["volume", "particle"]].copy()
    merged["n_scint0_entered"] = sum(df["n_scint0_entered"] for df in dfs)
    merged["n_stopped_here"]   = sum(df["n_stopped_here"]   for df in dfs)
    merged["cond_stop_frac"]   = (
        merged["n_stopped_here"] / merged["n_scint0_entered"].replace(0, 1)
    )
    return merged


def merge_decay_secondary_stopping(run_dirs: list[Path]) -> pd.DataFrame:
    """
    Sum n_entered and n_stopped per (volume, particle) across runs,
    recompute stop_frac and stop_frac_vs_total.

    stop_frac         = n_stopped / n_entered   (local)
    stop_frac_vs_total = n_stopped / Σ_all n_entered[kAll]
                         (absolute probability of ending up in vol X)
    """
    dfs = _load(run_dirs, "decay_secondary_stopping.csv")
    if not dfs:
        return pd.DataFrame()

    # Align on (volume, particle) — fill missing rows with 0
    base = dfs[0][["volume", "particle"]].copy()
    merged = base.copy()
    merged["n_entered"] = sum(
        df.set_index(["volume", "particle"])["n_entered"]
          .reindex(base.set_index(["volume", "particle"]).index, fill_value=0)
          .values
        for df in dfs
    )
    merged["n_stopped"] = sum(
        df.set_index(["volume", "particle"])["n_stopped"]
          .reindex(base.set_index(["volume", "particle"]).index, fill_value=0)
          .values
        for df in dfs
    )

    merged["stop_frac"] = merged["n_stopped"] / merged["n_entered"].replace(0, 1)

    # Total crossings = sum of n_entered for "all" particle bin across all vols
    total_all = merged.loc[merged["particle"] == "all", "n_entered"].sum()
    merged["stop_frac_vs_total"] = merged["n_stopped"] / max(total_all, 1)

    return merged[merged["n_entered"] > 0].reset_index(drop=True)


def merge_decay_summary(run_dirs: list[Path], n_events_total: int) -> pd.DataFrame:
    """
    Sum n_decay_mum and n_decay_mup per volume across runs,
    recompute per-event rates and total.
    The TOTAL row is reconstructed from the per-volume sums.
    """
    dfs = _load(run_dirs, "decay_summary.csv")
    if not dfs:
        return pd.DataFrame()

    dfs_novol = [df[df["volume"] != "TOTAL"].copy() for df in dfs]
    if not any(len(d) > 0 for d in dfs_novol):
        return pd.DataFrame()

    merged = dfs_novol[0][["volume"]].copy()
    merged["n_decay_mum"] = sum(df["n_decay_mum"] for df in dfs_novol)
    merged["n_decay_mup"] = sum(df["n_decay_mup"] for df in dfs_novol)
    merged["n_decay_total"] = merged["n_decay_mum"] + merged["n_decay_mup"]

    n = max(n_events_total, 1)
    merged["decay_rate_mum_per_event"] = merged["n_decay_mum"] / n
    merged["decay_rate_mup_per_event"] = merged["n_decay_mup"] / n

    total_row = pd.DataFrame([{
        "volume":                   "TOTAL",
        "n_decay_mum":              merged["n_decay_mum"].sum(),
        "n_decay_mup":              merged["n_decay_mup"].sum(),
        "n_decay_total":            merged["n_decay_total"].sum(),
        "decay_rate_mum_per_event": merged["n_decay_mum"].sum() / n,
        "decay_rate_mup_per_event": merged["n_decay_mup"].sum() / n,
    }])
    return pd.concat([merged, total_row], ignore_index=True)


def merge_coincidence_summary(run_dirs: list[Path],
                               n_events_total: int) -> str:
    """
    Merge coincidence_summary.csv across runs.

    coincidences.csv is concatenated as a raw event file (via
    _concat_with_offset in merge_runs).  The summary file contains
    aggregate counts, so we re-derive it by summing n_events columns
    across all runs for each section, then recompute rates and
    efficiencies from the merged totals.

    Returns the merged file content as a string ready to write.
    """
    import io as _io

    header_map = {
        "class,n_events,rate_per_event":                     "classes",
        "pair,n_events,rate_per_event":                      "pairs",
        "plate,n_trigger,n_4fold,efficiency,efficiency_err": "efficiency",
        "plate,n_coinc_events,mean_edep_MeV":                "edep",
        "mask,label,n_events,rate_per_event,n_with_al":      "patterns",
    }

    def _parse(path):
        sections = {}
        current, buf = None, []
        def _flush():
            if current and buf:
                try:
                    sections[current] = pd.read_csv(
                        _io.StringIO("\n".join(buf)))
                except Exception:
                    pass
        for line in path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s in header_map:
                _flush(); current = header_map[s]; buf = [s]
            elif current:
                buf.append(s)
        _flush()
        return sections

    all_sections: list[dict] = []
    for d in run_dirs:
        p = d / "coincidence_summary.csv"
        if p.exists():
            all_sections.append(_parse(p))

    if not all_sections:
        return ""

    n = max(n_events_total, 1)
    out_lines = [
        f"# Merged coincidence summary ({len(all_sections)} runs)",
        f"# Total events: {n_events_total}",
        "#",
    ]

    # ── classes ──────────────────────────────────────────────────────────────
    cls_dfs = [s["classes"] for s in all_sections if "classes" in s]
    if cls_dfs:
        merged = cls_dfs[0][["class"]].copy()
        merged["n_events"] = sum(df["n_events"] for df in cls_dfs)
        merged["rate_per_event"] = merged["n_events"] / n
        out_lines.append("class,n_events,rate_per_event")
        for _, row in merged.iterrows():
            out_lines.append(f"{row['class']},{int(row['n_events'])},"
                             f"{row['rate_per_event']:.6g}")
        out_lines.append("#")

    # ── pairs ─────────────────────────────────────────────────────────────────
    pair_dfs = [s["pairs"] for s in all_sections if "pairs" in s]
    if pair_dfs:
        merged = pair_dfs[0][["pair"]].copy()
        merged["n_events"] = sum(df["n_events"] for df in pair_dfs)
        merged["rate_per_event"] = merged["n_events"] / n
        out_lines.append("pair,n_events,rate_per_event")
        for _, row in merged.iterrows():
            out_lines.append(f"{row['pair']},{int(row['n_events'])},"
                             f"{row['rate_per_event']:.6g}")
        out_lines.append("#")

    # ── efficiency ────────────────────────────────────────────────────────────
    eff_dfs = [s["efficiency"] for s in all_sections if "efficiency" in s]
    if eff_dfs:
        merged = eff_dfs[0][["plate"]].copy()
        merged["n_trigger"] = sum(df["n_trigger"] for df in eff_dfs)
        merged["n_4fold"]   = sum(df["n_4fold"]   for df in eff_dfs)
        merged["efficiency"] = merged["n_4fold"] / merged["n_trigger"].replace(0, 1)
        merged["efficiency_err"] = np.sqrt(
            merged["efficiency"] * (1 - merged["efficiency"]) /
            merged["n_trigger"].replace(0, 1))
        out_lines.append("plate,n_trigger,n_4fold,efficiency,efficiency_err")
        for _, row in merged.iterrows():
            out_lines.append(
                f"{row['plate']},{int(row['n_trigger'])},{int(row['n_4fold'])},"
                f"{row['efficiency']:.6g},{row['efficiency_err']:.6g}")
        out_lines.append("#")

    # ── mean edep ─────────────────────────────────────────────────────────────
    edep_dfs = [s["edep"] for s in all_sections if "edep" in s]
    if edep_dfs:
        merged = edep_dfs[0][["plate"]].copy()
        merged["n_coinc_events"] = sum(df["n_coinc_events"] for df in edep_dfs)
        # Weighted mean: sum(n_i * mean_i) / sum(n_i)
        weighted = sum(df["n_coinc_events"] * df["mean_edep_MeV"]
                       for df in edep_dfs)
        merged["mean_edep_MeV"] = weighted / merged["n_coinc_events"].replace(0, 1)
        out_lines.append("plate,n_coinc_events,mean_edep_MeV")
        for _, row in merged.iterrows():
            out_lines.append(f"{row['plate']},{int(row['n_coinc_events'])},"
                             f"{row['mean_edep_MeV']:.6g}")
        out_lines.append("#")

    # ── patterns ──────────────────────────────────────────────────────────────
    pat_dfs = [s["patterns"] for s in all_sections if "patterns" in s]
    if pat_dfs:
        merged = pat_dfs[0][["mask", "label"]].copy()
        merged["n_events"]      = sum(df["n_events"]  for df in pat_dfs)
        merged["n_with_al"]     = sum(df["n_with_al"] for df in pat_dfs)
        merged["rate_per_event"] = merged["n_events"] / n
        merged = merged[merged["n_events"] > 0]
        out_lines.append("mask,label,n_events,rate_per_event,n_with_al")
        for _, row in merged.iterrows():
            out_lines.append(
                f"{int(row['mask'])},{row['label']},{int(row['n_events'])},"
                f"{row['rate_per_event']:.6g},{int(row['n_with_al'])}")

    return "\n".join(out_lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main merge driver
# ──────────────────────────────────────────────────────────────────────────────

def merge_runs(run_dirs: list[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total, per_run = count_events(run_dirs)
    print(f"Found {len(run_dirs)} runs, {n_total:,} total events:")
    for d, n in zip(run_dirs, per_run):
        print(f"  {d}  →  {n:,} events")
    print()

    # ── Summary files ────────────────────────────────────────────────────────
    edep = merge_edep_summary(run_dirs, n_total)
    if not edep.empty:
        edep.to_csv(out_dir / "edep_summary.csv", index=False)
        print(f"  → {out_dir}/edep_summary.csv")

    stop = merge_stopping(run_dirs, n_total)
    if not stop.empty:
        stop.to_csv(out_dir / "stopping.csv", index=False)
        print(f"  → {out_dir}/stopping.csv")

    cond = merge_stopping_conditional(run_dirs)
    if not cond.empty:
        cond.to_csv(out_dir / "stopping_conditional.csv", index=False)
        print(f"  → {out_dir}/stopping_conditional.csv")

    flu = merge_fluence(run_dirs, n_total)
    if not flu.empty:
        flu.to_csv(out_dir / "fluence.csv", index=False)
        print(f"  → {out_dir}/fluence.csv")

    # ── entry_ke: full concatenation ─────────────────────────────────────────
    # New format uses volIdx (0–4); old format used plateIdx (0–3).
    # _concat_with_offset handles both since it only offsets eventID.
    ek = _concat_with_offset(run_dirs, "entry_ke.csv")
    if not ek.empty:
        # Normalise column name for the merged file
        if "plateIdx" in ek.columns and "volIdx" not in ek.columns:
            ek = ek.rename(columns={"plateIdx": "volIdx"})
        ek.to_csv(out_dir / "entry_ke.csv", index=False)
        n_scint = (ek["volIdx"] <= 3).sum() if "volIdx" in ek.columns else len(ek)
        n_al    = (ek["volIdx"] == 4).sum()  if "volIdx" in ek.columns else 0
        print(f"  → {out_dir}/entry_ke.csv  "
              f"({len(ek):,} records,  scint {n_scint:,}  Al {n_al:,})")

    # ── gun_energy: full concatenation ───────────────────────────────────────
    gun = _concat_with_offset(run_dirs, "gun_energy.csv")
    if not gun.empty:
        gun.to_csv(out_dir / "gun_energy.csv", index=False)
        n_mup = (gun["pdg"] == -13).sum()
        n_mum = (gun["pdg"] ==  13).sum()
        print(f"  → {out_dir}/gun_energy.csv  "
              f"({len(gun):,} records,  μ+ {n_mup:,}  μ− {n_mum:,}  "
              f"ratio {n_mup/max(n_mum,1):.3f})")

    # ── decay.csv: full concatenation ────────────────────────────────────────
    dec = _concat_with_offset(run_dirs, "decay.csv")
    if not dec.empty:
        dec.to_csv(out_dir / "decay.csv", index=False)
        n_al = (dec["volume"] == "Al_plate").sum() if "volume" in dec.columns else 0
        print(f"  → {out_dir}/decay.csv  "
              f"({len(dec):,} total decays,  {n_al:,} in Al plate)")
    else:
        print(f"  → decay.csv: no decay records found across runs "
              f"(expected for E > 100 MeV source)")

    # ── decay_summary: summed per-volume counts ───────────────────────────────
    dec_sum = merge_decay_summary(run_dirs, n_total)
    if not dec_sum.empty:
        dec_sum.to_csv(out_dir / "decay_summary.csv", index=False)
        total_row = dec_sum[dec_sum["volume"] == "TOTAL"]
        n_tot_dec = int(total_row["n_decay_total"].values[0]) \
                    if not total_row.empty else 0
        print(f"  → {out_dir}/decay_summary.csv  ({n_tot_dec:,} total decays)")

    # ── decay_secondary_entry_ke: full concatenation ─────────────────────────
    dec_sec_ke = _concat_with_offset(run_dirs, "decay_secondary_entry_ke.csv")
    if not dec_sec_ke.empty:
        dec_sec_ke.to_csv(out_dir / "decay_secondary_entry_ke.csv", index=False)
        vols = dec_sec_ke["volume"].unique().tolist() if "volume" in dec_sec_ke.columns else []
        print(f"  → {out_dir}/decay_secondary_entry_ke.csv  "
              f"({len(dec_sec_ke):,} records,  "
              f"volumes: {', '.join(sorted(vols))})")
    else:
        print("  → decay_secondary_entry_ke.csv: no records "
              "(no Al-plate decays recorded)")
    dec_sec = merge_decay_secondary_stopping(run_dirs)
    if not dec_sec.empty:
        dec_sec.to_csv(out_dir / "decay_secondary_stopping.csv", index=False)
        total_stopped = int(dec_sec.loc[dec_sec["particle"] == "all",
                                        "n_stopped"].sum())
        print(f"  → {out_dir}/decay_secondary_stopping.csv  "
              f"({total_stopped:,} total secondary stops)")
    else:
        print("  → decay_secondary_stopping.csv: no data "
              "(no Al-plate decays recorded)")

    # ── coincidences.csv: full concatenation ─────────────────────────────────
    coinc = _concat_with_offset(run_dirs, "coincidences.csv")
    if not coinc.empty:
        coinc.to_csv(out_dir / "coincidences.csv", index=False)
        n4 = (coinc["n_fired"] == 4).sum() if "n_fired" in coinc.columns else 0
        print(f"  → {out_dir}/coincidences.csv  "
              f"({len(coinc):,} triggered events,  {n4:,} 4-fold)")
    else:
        print("  → coincidences.csv: no triggered events found")

    # ── coincidence_summary.csv: re-derived from merged counts ───────────────
    coinc_sum_text = merge_coincidence_summary(run_dirs, n_total)
    if coinc_sum_text:
        (out_dir / "coincidence_summary.csv").write_text(coinc_sum_text)
        print(f"  → {out_dir}/coincidence_summary.csv")
    else:
        print("  → coincidence_summary.csv: no data found")

    # ── Per-event / step files: copy from first run that has them ────────────
    for fname in ["edep_per_event.csv", "steps.csv", "trajectories.csv"]:
        for d in run_dirs:
            src = d / fname
            if src.exists() and src.stat().st_size > 100:
                shutil.copy2(src, out_dir / fname)
                print(f"  → {out_dir}/{fname}  (copied from {d})")
                break

    print(f"\nDone. Run:  python plot_results.py --dir {out_dir} --save")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge summary CSVs from parallel CosmicMuonSim runs.")
    parser.add_argument("dirs", nargs="+", type=Path,
                        help="paths to results/ directories from each run")
    parser.add_argument("-o", "--output", type=Path, default=Path("merged_results"),
                        help="output directory (default: merged_results)")
    args = parser.parse_args()

    valid = []
    for d in args.dirs:
        if d.is_dir():
            valid.append(d)
        else:
            print(f"  warning: '{d}' not found, skipping.")

    if len(valid) < 2:
        print("Error: need at least 2 valid run directories to merge.")
        sys.exit(1)

    merge_runs(valid, args.output)


if __name__ == "__main__":
    main()