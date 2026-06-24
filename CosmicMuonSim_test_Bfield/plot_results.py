#!/usr/bin/env python3
"""
plot_results.py — Publication-quality plots for CosmicMuonSim output.

  - Loads entry_ke.csv for true entry-KE distributions (no event cap,
    one record per track crossing per scintillator plate).
  - Loads gun_energy.csv for source validation (pure Gaisser sampler output,
    zero physics interaction — recorded directly from PrimaryGeneratorAction).
  - fig_entry_ke_distributions: per-plate muon entry KE with full statistics.
  - fig_gun_energy: three-panel source validation figure.
  - fig_ke_spectra: step-level all-species KE spectra (secondaries etc.)
  - fig_stopping: two panels —
      left:  stop_frac        = n_stopped[vol] / n_entered[vol]  (local)
      right: cond_stop_frac   = n_stopped_here / n_scint0_entered (conditional,
             from stopping_conditional.csv — tracks that entered Scint_0 first)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

VOLUME_ORDER = [
    "Scint_0", "Scint_1", "Al_plate", "Scint_2", "Scint_3",
    "Cu_walls", "Air_hole", "Ceiling", "Roof",
]
VOL_LABELS = {
    "Scint_0": "Scint 0",  "Scint_1": "Scint 1",
    "Scint_2": "Scint 2",  "Scint_3": "Scint 3",
    "Al_plate": "Al plate", "Cu_walls": "Cu walls",
    "Air_hole": "Air gap",  "Ceiling": "Ceiling",
    "Roof": "Roof",         "World": "World",
}
SCINT_LABELS = {
    0: "Scint 0  (+74 mm, top)",
    1: "Scint 1  (+54 mm)",
    2: "Scint 2  (−56 mm)",
    3: "Scint 3  (−76 mm, bottom)",
}
PARTICLES = ["mu-", "mu+", "e-", "e+", "gamma", "other"]
PART_COLOURS = {
    "mu-":    "#d62728",
    "mu+":    "#1f77b4",
    "e-":     "#2ca02c",
    "e+":     "#ff7f0e",
    "gamma":  "#9467bd",
    "other":  "#8c564b",
    "all":    "#333333",
}
VOL_COLOURS = {
    "Scint_0": "#66c2a5", "Scint_1": "#a6d854",
    "Scint_2": "#66c2a5", "Scint_3": "#a6d854",
    "Al_plate": "#8da0cb", "Cu_walls": "#e78ac3",
    "Air_hole": "#e5e5e5", "Ceiling":  "#b3b3b3",
    "Roof":     "#969696", "World":    "#f0f0f0",
}
SCINT_COLOURS = ["#2E86AB", "#27AE60", "#E67E22", "#C0392B"]

VALID_VOLUMES = {
    "Scint_0", "Scint_1", "Scint_2", "Scint_3",
    "Al_plate", "Cu_walls", "Air_hole",
    "Ceiling", "Roof", "World",
}


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 9, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
        "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })


def _clean(text: str) -> str:
    return text.replace("..", ".")


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_edep_summary(d):
    p = d / "edep_summary.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    if "total_edep_MeV" in df.columns and df["total_edep_MeV"].sum() == 0:
        print("  warning: edep_summary.csv all zeros; will recompute.")
        return pd.DataFrame()
    return df


def load_stopping(d):
    p = d / "stopping.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    if "n_entered" in df.columns and df["n_entered"].sum() == 0:
        print("  warning: stopping.csv all zeros.")
        return pd.DataFrame()
    return df


def load_stopping_conditional(d):
    p = d / "stopping_conditional.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    if "n_scint0_entered" in df.columns and df["n_scint0_entered"].sum() == 0:
        print("  warning: stopping_conditional.csv all zeros.")
        return pd.DataFrame()
    return df


def load_fluence(d):
    p = d / "fluence.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    if "total_steplen_mm" in df.columns and df["total_steplen_mm"].sum() == 0:
        print("  warning: fluence.csv all zeros.")
        return pd.DataFrame()
    return df


def load_edep_per_event(d):
    p = d / "edep_per_event.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")
    if "volume" in df.columns:
        before = len(df)
        df = df[df["volume"].isin(VALID_VOLUMES)].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  warning: Dropped {dropped} corrupted rows from edep_per_event.csv")
    return df


def load_steps(d):
    p = d / "steps.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")
    rn = {"edep_MeV": "edep", "stepLen_mm": "step_len", "KE_MeV": "KE",
          "x_mm": "x", "y_mm": "y", "z_mm": "z", "time_ns": "time"}
    df = df.rename(columns=rn)
    for col in ["plateIdx", "edep", "KE", "x", "y", "z", "pdg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plateIdx", "edep", "KE", "x", "y", "z"])
    df = df[df["plateIdx"].between(0, 3)].copy()
    df["plateIdx"] = df["plateIdx"].astype(int)
    return df


def load_trajectories(d):
    p = d / "trajectories.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")
    rn = {"edep_MeV": "edep", "stepLen_mm": "step_len", "KE_MeV": "KE",
          "x_mm": "x", "y_mm": "y", "z_mm": "z", "time_ns": "time"}
    df = df.rename(columns=rn)
    for col in ["trackID", "parentID", "pdg", "edep", "KE", "x", "y", "z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trackID", "x", "y", "z"])
    return df


def load_entry_ke(d):
    """
    Load entry_ke.csv — muon KE at scintillator AND Al plate boundaries.

    Old format: eventID, plateIdx, pdg, trackID, ke_MeV  (scint only, 0–3)
    New format: eventID, volIdx,   pdg, trackID, ke_MeV  (scint 0–3 + Al=4)

    Returns a DataFrame with column 'volIdx' in both cases.
    Filters to only muon PDGs (±13).
    """
    p = d / "entry_ke.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")

    # Normalise column name
    if "plateIdx" in df.columns and "volIdx" not in df.columns:
        df = df.rename(columns={"plateIdx": "volIdx"})

    for col in ["volIdx", "pdg", "trackID", "ke_MeV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["volIdx", "ke_MeV"])
    # Accept 0–4: scint 0–3 plus Al plate (4)
    df = df[df["volIdx"].between(0, 4)].copy()
    df["volIdx"] = df["volIdx"].astype(int)

    n_scint = (df["volIdx"] <= 3).sum()
    n_al    = (df["volIdx"] == 4).sum()
    print(f"  entry_ke: {len(df):,} muon entries  "
          f"({n_scint:,} scint,  {n_al:,} Al plate)")
    return df


def load_decay_secondary_entry_ke(d):
    """
    Load decay_secondary_entry_ke.csv — entry KE for Al-plate decay
    secondaries crossing into any volume.
    Columns: eventID, volume, pdg, trackID, ke_MeV
    """
    p = d / "decay_secondary_entry_ke.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")
    for col in ["pdg", "trackID", "ke_MeV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["volume", "ke_MeV"])
    vols_present = df["volume"].unique().tolist()
    print(f"  decay_secondary_entry_ke: {len(df):,} records  "
          f"(volumes: {', '.join(sorted(vols_present))})")
    return df


def load_gun_energy(d):
    """
    Load gun_energy.csv — pure Gaisser sampler output, one row per primary.
    Columns: eventID, pdg, energy_MeV, cosTheta
    Written directly from PrimaryGeneratorAction before any physics.
    """
    p = d / "gun_energy.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())), on_bad_lines="skip")
    for col in ["pdg", "energy_MeV", "cosTheta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["pdg", "energy_MeV", "cosTheta"])
    print(f"  gun_energy: {len(df):,} records  "
          f"({(df['pdg']==-13).sum():,} μ+,  {(df['pdg']==13).sum():,} μ−)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fallback deriver
# ──────────────────────────────────────────────────────────────────────────────

def derive_edep_summary(epe):
    if epe.empty: return pd.DataFrame()
    part_cols = {"mu-_MeV": "mu-", "mu+_MeV": "mu+", "e-_MeV": "e-",
                 "e+_MeV": "e+", "gamma_MeV": "gamma",
                 "other_MeV": "other", "all_MeV": "all"}
    n_events = epe["eventID"].nunique()
    rows = []
    for vol in VALID_VOLUMES:
        sub = epe[epe["volume"] == vol]
        for col_csv, part_name in part_cols.items():
            if col_csv not in sub.columns: continue
            total = sub[col_csv].sum()
            rows.append({"volume": vol, "particle": part_name,
                         "total_edep_MeV": total,
                         "mean_edep_MeV": total / n_events if n_events > 0 else 0})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def fig_edep_by_volume(edep, out, fmt):
    if edep.empty: return
    vols = [v for v in VOLUME_ORDER if v in edep["volume"].values]
    df = edep[(edep["volume"].isin(vols)) & (edep["particle"] == "all")]
    df = df.set_index("volume").loc[vols]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(vols))
    colours = [VOL_COLOURS.get(v, "#ccc") for v in vols]
    bars = ax.bar(x, df["total_edep_MeV"], color=colours, edgecolor="0.3",
                  linewidth=0.6, zorder=3)
    for bar, val in zip(bars, df["total_edep_MeV"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols], rotation=30, ha="right")
    ax.set_ylabel("Total energy deposit  [MeV]")
    ax.set_title("Energy deposition by detector component (all particles)")
    ax.grid(axis="y", zorder=0)
    ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, out, "edep_by_volume", fmt)


def fig_edep_stacked(edep, out, fmt):
    if edep.empty: return
    vols = [v for v in VOLUME_ORDER if v in edep["volume"].values]
    parts = [p for p in PARTICLES if p in edep["particle"].values]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(vols))
    bottoms = np.zeros(len(vols))
    for part in parts:
        vals = []
        for v in vols:
            row = edep[(edep["volume"] == v) & (edep["particle"] == part)]
            vals.append(row["total_edep_MeV"].values[0] if len(row) else 0.)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, label=part, width=0.7,
               color=PART_COLOURS[part], edgecolor="white", linewidth=0.4, zorder=3)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols], rotation=30, ha="right")
    ax.set_ylabel("Total energy deposit  [MeV]")
    ax.set_title("Energy deposition by component — particle breakdown")
    ax.legend(ncol=3, loc="upper right", frameon=True)
    ax.grid(axis="y", zorder=0)
    ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, out, "edep_stacked", fmt)


def fig_edep_distributions(epe, out, fmt):
    if epe.empty: return
    scints = [s for s in ["Scint_0", "Scint_1", "Scint_2", "Scint_3"]
              if s in epe["volume"].values]
    if not scints: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.ravel()
    for i, vol in enumerate(scints):
        ax = axes[i]
        sub = epe[epe["volume"] == vol]
        total = sub["all_MeV"]
        total = total[total > 0]
        if total.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.5")
            ax.set_title(VOL_LABELS.get(vol, vol))
            continue
        ax.hist(total, bins=60, color=VOL_COLOURS.get(vol, "#ccc"),
                edgecolor="0.3", linewidth=0.4, alpha=0.85, zorder=3)
        ax.axvline(total.mean(), color="k", ls="--", lw=1,
                   label=f"mean = {total.mean():.3f} MeV")
        ax.axvline(total.median(), color="0.4", ls=":", lw=1,
                   label=f"median = {total.median():.3f} MeV")
        ax.set_xlabel("Energy deposit per event  [MeV]")
        ax.set_ylabel("Events")
        ax.set_title(VOL_LABELS.get(vol, vol))
        ax.legend(fontsize=8)
        ax.grid(axis="y", zorder=0)
    fig.suptitle("Per-event energy deposit distributions (non-zero events)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "edep_distributions", fmt)


def _gaisser_eq3(E_GeV, cos_theta):
    """
    Modified Gaisser parametrization — Eq. 3 of Guan et al. (arXiv:1509.06176).

    dI/dE = 0.14 * [E/GeV * (1 + 3.64/(E * cosθ*^1.29))]^-2.7
              * [1/(1 + 1.1*E*cosθ*/115) + 0.054/(1 + 1.1*E*cosθ*/850)]

    where cosθ* is the Earth-curvature-corrected zenith angle (Eq. 2).
    Returns the differential flux shape (arbitrary normalisation).
    """
    # ── Eq. 2: cosθ* ────────────────────────────────────────────────────────
    P1, P2, P3, P4, P5 = 0.102573, -0.068287, 0.958633, 0.0407253, 0.817285
    c = np.asarray(cos_theta, dtype=float)
    num = c**2 + P1**2 + P2 * c**P3 + P4 * c**P5
    den = 1.0 + P1**2 + P2 + P4
    cos_star = np.sqrt(np.maximum(num / den, 0.0))
    cos_star = np.maximum(cos_star, 1e-6)

    E = np.asarray(E_GeV, dtype=float)

    # ── Low-energy correction term ───────────────────────────────────────────
    low_e = (1.0 + 3.64 / (E * cos_star**1.29)) ** -2.7

    # ── Gaisser bracket (pion + kaon decay) ─────────────────────────────────
    a = 1.1 * E * cos_star
    bracket = 1.0 / (1.0 + a / 115.0) + 0.054 / (1.0 + a / 850.0)

    return 0.14 * E**-2.7 * low_e * bracket


def _gaisser_dNdE(E_GeV_centres, cos_grid):
    """
    dN/dE ∝ ∫₀¹ dI/dE(E, cosθ) d(cosθ)
    Numerically integrated over the cosθ grid using the trapezoid rule.
    """
    # Shape: (n_E, n_cos) → integrate over cos axis
    flux = _gaisser_eq3(E_GeV_centres[:, None], cos_grid[None, :])
    return np.trapezoid(flux, cos_grid, axis=1)


def _gaisser_dNdcos(cos_centres, E_GeV_grid):
    """
    dN/d(cosθ) ∝ ∫_{Emin}^{Emax} dI/dE(E, cosθ) dE
    Numerically integrated over the energy grid using the trapezoid rule.
    """
    flux = _gaisser_eq3(E_GeV_grid[None, :], cos_centres[:, None])
    return np.trapezoid(flux, E_GeV_grid, axis=1)


def fig_gun_energy(gun, out, fmt):
    """
    Three-panel source validation from gun_energy.csv.

    All data comes directly from the Gaisser sampler — no physics has
    touched these values — so any deviation from the expected shapes is
    a bug in the sampling, not a physical effect.

    Left   — dN/dE energy spectrum (log-log).
              Reference: Eq. 3 integrated over cosθ ∈ [0, 1].
    Centre — cosθ distribution (linear).
              Reference: Eq. 3 integrated over E ∈ [Emin, Emax].
    Right  — μ+/μ− charge ratio vs energy.
              Expected flat line at 1.27 (Gaisser charge ratio).
    """
    if gun.empty:
        print("  skipping gun energy plot: gun_energy.csv not available.")
        return

    mum = gun[gun["pdg"] ==  13]   # μ−  (PDG +13)
    mup = gun[gun["pdg"] == -13]   # μ+  (PDG −13)

    fig, (ax_e, ax_cos, ax_ratio) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Integration grids (fine, for smooth reference curves) ───────────────
    E_min_GeV = 0.1
    E_max_GeV = 100.0
    cos_grid  = np.linspace(0.001, 1.0, 300)   # avoid exactly 0 (cos 90°)
    E_grid    = np.logspace(np.log10(E_min_GeV), np.log10(E_max_GeV), 400)

    # ── Left: energy spectrum dN/dE ─────────────────────────────────────────
    e_min = max(gun["energy_MeV"].min(), 100.0)
    e_max = gun["energy_MeV"].max()
    bins_e    = np.logspace(np.log10(e_min), np.log10(e_max), 60)
    widths    = np.diff(bins_e)
    centres_e = 0.5 * (bins_e[:-1] + bins_e[1:])

    # Combined filled histogram (normalised to dN/dE)
    all_counts, _ = np.histogram(gun["energy_MeV"], bins=bins_e)
    all_dndE = all_counts / widths
    ax_e.fill_between(centres_e, all_dndE, step="mid",
                      color="0.6", alpha=0.35, zorder=2,
                      label=f"μ+μ−  (N={len(gun):,})")

    # Individual species as step lines
    for label, df, col in [("μ−", mum, PART_COLOURS["mu-"]),
                            ("μ+", mup, PART_COLOURS["mu+"])]:
        if df.empty: continue
        counts, _ = np.histogram(df["energy_MeV"], bins=bins_e)
        ax_e.step(centres_e, counts / widths, where="mid",
                  color=col, lw=1.5, zorder=3, label=f"{label}  (N={len(df):,})")

    # Reference: Eq. 3 marginalised over cosθ, normalised to combined histogram
    ref_dndE = _gaisser_dNdE(centres_e / 1000.0, cos_grid)
    norm_e = np.sum(all_dndE * widths) / np.sum(ref_dndE * widths)
    ax_e.plot(centres_e, norm_e * ref_dndE, "k-", lw=1.6, alpha=0.85,
              label="Eq. 3  (∫ over cosθ)")

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("Kinetic energy  [MeV]")
    ax_e.set_ylabel("dN/dE  [counts / MeV]")
    ax_e.set_title("Energy spectrum\n(solid black = Eq. 3 integrated over cosθ)")
    ax_e.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_e.legend(fontsize=7.5)
    ax_e.grid(True, which="both", alpha=0.2)

    # ── Centre: cosθ distribution ────────────────────────────────────────────
    bins_cos    = np.linspace(0, 1, 50)
    centres_cos = 0.5 * (bins_cos[:-1] + bins_cos[1:])
    w_cos       = np.diff(bins_cos)

    # Combined filled histogram
    all_cos_counts, _ = np.histogram(gun["cosTheta"], bins=bins_cos)
    all_dndcos = all_cos_counts / w_cos
    ax_cos.fill_between(centres_cos, all_dndcos, step="mid",
                        color="0.6", alpha=0.35, zorder=2,
                        label=f"μ+μ−  (N={len(gun):,})")

    # Individual species as step lines
    for label, df, col in [("μ−", mum, PART_COLOURS["mu-"]),
                            ("μ+", mup, PART_COLOURS["mu+"])]:
        if df.empty: continue
        counts, _ = np.histogram(df["cosTheta"], bins=bins_cos)
        ax_cos.step(centres_cos, counts / w_cos, where="mid",
                    color=col, lw=1.5, zorder=3, label=f"{label}  (N={len(df):,})")

    # Reference: Eq. 3 marginalised over E, normalised to combined histogram
    ref_dndcos = _gaisser_dNdcos(centres_cos, E_grid)
    norm_cos = np.sum(all_dndcos * w_cos) / np.sum(ref_dndcos * w_cos)
    ax_cos.plot(centres_cos, norm_cos * ref_dndcos, "k-", lw=1.6, alpha=0.85,
                label="Eq. 3  (∫ over E)")

    ax_cos.set_xlabel(r"cos $\theta$")
    ax_cos.set_ylabel(r"dN / d(cos$\theta$)  [counts / bin width]")
    ax_cos.set_title("Zenith angle distribution\n"
                     "(solid black = Eq. 3 integrated over E)")
    ax_cos.legend(fontsize=7.5)
    ax_cos.grid(True, alpha=0.2)

    # ── Right: μ+/μ− charge ratio vs energy ─────────────────────────────────
    mum_counts, _ = np.histogram(mum["energy_MeV"], bins=bins_e)
    mup_counts, _ = np.histogram(mup["energy_MeV"], bins=bins_e)

    mask = (mum_counts > 5) & (mup_counts > 5)
    if mask.sum() > 3:
        ratio     = mup_counts[mask] / mum_counts[mask]
        ratio_err = ratio * np.sqrt(1.0 / mup_counts[mask] +
                                    1.0 / mum_counts[mask])
        ax_ratio.errorbar(centres_e[mask], ratio, yerr=ratio_err,
                          fmt="o", ms=4, color="#444",
                          elinewidth=0.8, capsize=2,
                          label="μ+/μ−  (binned)")
        ax_ratio.axhline(1.27, color=PART_COLOURS["mu-"], lw=1.5, ls="--",
                         label="Expected  1.27")
        ax_ratio.axhline(1.00, color="0.6", lw=0.8, ls=":")

        r_global = len(mup) / max(len(mum), 1)
        ax_ratio.text(0.05, 0.95,
                      f"Global μ+/μ−  =  {r_global:.3f}\n"
                      f"(expected 1.27,  set 56/44 split)",
                      transform=ax_ratio.transAxes,
                      va="top", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                alpha=0.8, ec="0.7"))
        ax_ratio.set_ylim(0, 3.0)
        ax_ratio.legend(fontsize=8)
    else:
        ax_ratio.text(0.5, 0.5, "Insufficient statistics",
                      ha="center", va="center", transform=ax_ratio.transAxes,
                      color="0.5", fontsize=11)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Kinetic energy  [MeV]")
    ax_ratio.set_ylabel("μ+ / μ−  ratio")
    ax_ratio.set_title("Charge ratio vs energy\n(should be flat at 1.27)")
    ax_ratio.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_ratio.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "Gun source validation  —  gun_energy.csv\n"
        "(pure Gaisser sampler output, recorded before any physics interaction)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out, "gun_energy_validation", fmt)


def fig_entry_ke_distributions(entry_ke, out, fmt):
    """
    Entry kinetic energy for muons at each scintillator plate (volIdx 0–3).
    Y-axis is dN/dE [tracks / MeV]. Al plate handled in fig_al_entry_ke.
    """
    if entry_ke.empty:
        print("  skipping entry-KE plot: entry_ke.csv not available.")
        return

    muons = entry_ke[entry_ke["pdg"].isin([13, -13])].copy()
    muons = muons[muons["volIdx"] <= 3]
    if muons.empty:
        print("  skipping entry-KE plot: no scint muon records in entry_ke.csv.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx in range(4):
        ax = axes[idx]
        data_all = muons[muons["volIdx"] == idx]["ke_MeV"]
        data_mum = muons[(muons["volIdx"] == idx) & (muons["pdg"] ==  13)]["ke_MeV"]
        data_mup = muons[(muons["volIdx"] == idx) & (muons["pdg"] == -13)]["ke_MeV"]

        if data_all.empty:
            ax.text(0.5, 0.5, "No muon entries\n(muons did not reach this plate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5", fontsize=11)
            ax.set_title(SCINT_LABELS[idx])
            continue

        ke_min = max(data_all.min(), 1.0)
        ke_max = data_all.max()
        bins    = np.logspace(np.log10(ke_min), np.log10(ke_max), 70)
        widths  = np.diff(bins)
        centres = 0.5 * (bins[:-1] + bins[1:])

        # Combined filled area
        counts_all, _ = np.histogram(data_all, bins=bins)
        ax.fill_between(centres, counts_all / widths, step="mid",
                        color=SCINT_COLOURS[idx], alpha=0.35, zorder=2,
                        label=f"μ+μ−  (N={len(data_all):,})")

        # Individual species as step lines
        for data, col, lbl in [(data_mum, PART_COLOURS["mu-"], f"μ−  (N={len(data_mum):,})"),
                                (data_mup, PART_COLOURS["mu+"], f"μ+  (N={len(data_mup):,})")]:
            if data.empty: continue
            counts, _ = np.histogram(data, bins=bins)
            mask = counts > 0
            ax.step(centres[mask], (counts / widths)[mask], where="mid",
                    lw=1.5, color=col, zorder=3, label=lbl)

        # Median and mean as vertical lines
        med = data_all.median()
        ax.axvline(med, color="0.2", ls="--", lw=1.2,
                   label=f"median  {med:,.0f} MeV")
        ax.axvline(data_all.mean(), color="0.2", ls=":", lw=1.0,
                   label=f"mean  {data_all.mean():,.0f} MeV")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Entry kinetic energy  [MeV]")
        ax.set_ylabel("dN/dE  [tracks / MeV]")
        ax.set_title(SCINT_LABELS[idx])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, which="both", alpha=0.2)

        q10, q90 = data_all.quantile(0.10), data_all.quantile(0.90)
        stats_text = (f"10th: {q10:,.0f} MeV\n"
                      f"90th: {q90:,.0f} MeV\n"
                      f"max:  {data_all.max():,.0f} MeV")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="0.7"))

    fig.suptitle(
        "Muon entry kinetic energy at each scintillator plate\n"
        "(from entry_ke.csv — full-run statistics, true boundary-crossing KE)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "entry_ke_distributions", fmt)


def fig_entry_ke_overlay(entry_ke, out, fmt):
    """
    All four scintillator plates + Al plate overlaid on one log-log axes.
    Shows how the spectrum evolves as muons penetrate the detector stack.
    """
    if entry_ke.empty:
        return

    muons = entry_ke[entry_ke["pdg"].isin([13, -13])].copy()
    if muons.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    all_ke = muons["ke_MeV"]
    ke_min = max(all_ke.min(), 1.0)
    bins = np.logspace(np.log10(ke_min), np.log10(all_ke.max()), 80)

    # Scintillator plates
    for idx in range(4):
        data = muons[muons["volIdx"] == idx]["ke_MeV"]
        if data.empty:
            continue
        counts, edges = np.histogram(data, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centres, counts, where="mid",
                color=SCINT_COLOURS[idx], lw=1.8,
                label=f"{SCINT_LABELS[idx]}  (N={len(data):,})")

    # Al plate (volIdx = 4)
    al_data = muons[muons["volIdx"] == 4]["ke_MeV"]
    if not al_data.empty:
        counts, edges = np.histogram(al_data, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centres, counts, where="mid",
                color=VOL_COLOURS["Al_plate"], lw=2.2, ls="--",
                label=f"Al plate  (N={len(al_data):,})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Entry kinetic energy  [MeV]")
    ax.set_ylabel("Muon entries")
    ax.set_title("Muon entry KE — scintillators + Al plate overlaid\n"
                 "(low-energy tail filtered by upstream material)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    _save(fig, out, "entry_ke_overlay", fmt)


def fig_al_entry_ke(entry_ke, out, fmt):
    """
    Dedicated figure for muon KE at the Al plate entry (volIdx = 4).

    Two panels:
      Left  — KE spectrum with μ− and μ+ overlaid, compared to Scint_0
               spectrum (upstream reference) to show how much energy muons
               lose before reaching the Al absorber.
      Right — Cumulative fraction above a given KE threshold, useful for
               estimating what fraction of muons have enough range to stop
               in the plate.
    """
    if entry_ke.empty:
        print("  skipping Al entry KE: entry_ke.csv not available.")
        return

    al = entry_ke[(entry_ke["volIdx"] == 4) & entry_ke["pdg"].isin([13, -13])].copy()
    if al.empty:
        print("  skipping Al entry KE: no muon entries recorded at Al plate.")
        return

    s0 = entry_ke[(entry_ke["volIdx"] == 0) & entry_ke["pdg"].isin([13, -13])].copy()

    fig, (ax_spec, ax_cdf) = plt.subplots(1, 2, figsize=(13, 5))

    ke_min = 1.0
    ke_max = max(al["ke_MeV"].max(), s0["ke_MeV"].max() if not s0.empty else 1.0)
    bins   = np.logspace(np.log10(ke_min), np.log10(ke_max), 70)
    widths = np.diff(bins)
    centres = 0.5 * (bins[:-1] + bins[1:])

    # ── Left: dN/dE spectra ──────────────────────────────────────────────────
    # Al plate combined (filled)
    al_counts, _ = np.histogram(al["ke_MeV"], bins=bins)
    ax_spec.fill_between(centres, al_counts / widths, step="mid",
                         color=VOL_COLOURS["Al_plate"], alpha=0.4, zorder=2,
                         label=f"Al plate μ+μ−  (N={len(al):,})")

    for label, pdg_val, col in [("μ−", 13, PART_COLOURS["mu-"]),
                                  ("μ+", -13, PART_COLOURS["mu+"])]:
        sub = al[al["pdg"] == pdg_val]["ke_MeV"]
        if sub.empty: continue
        c, _ = np.histogram(sub, bins=bins)
        ax_spec.step(centres, c / widths, where="mid",
                     color=col, lw=1.5, zorder=3,
                     label=f"Al plate {label}  (N={len(sub):,})")

    # Scint_0 reference (dashed)
    if not s0.empty:
        s0_counts, _ = np.histogram(s0["ke_MeV"], bins=bins)
        # Normalise to same area as Al for shape comparison
        norm = (al_counts / widths).sum() / max((s0_counts / widths).sum(), 1)
        ax_spec.step(centres, norm * s0_counts / widths, where="mid",
                     color="0.3", lw=1.3, ls="--", zorder=2,
                     label=f"Scint 0 (rescaled, N={len(s0):,})")

    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("Entry kinetic energy  [MeV]")
    ax_spec.set_ylabel("dN/dE  [counts / MeV]")
    ax_spec.set_title("Muon KE at Al plate entry\n"
                       "(dashed = Scint 0 rescaled for shape comparison)")
    ax_spec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, which="both", alpha=0.2)

    # ── Right: cumulative fraction above threshold ────────────────────────────
    sorted_ke = np.sort(al["ke_MeV"].values)
    cdf_above = 1.0 - np.arange(1, len(sorted_ke) + 1) / len(sorted_ke)
    ax_cdf.plot(sorted_ke, cdf_above, color=VOL_COLOURS["Al_plate"],
                lw=2.0, label="Al plate  (all μ)")

    # Mark the approximate minimum KE to traverse 10 mm Al (~20 MeV for slow μ)
    ax_cdf.axvline(20, color="#c0392b", ls=":", lw=1.3,
                   label="~20 MeV  (min range ≈ plate thickness)")
    ax_cdf.axvline(100, color="0.5", ls=":", lw=1.0,
                   label="100 MeV  (source lower limit)")

    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Entry KE threshold  [MeV]")
    ax_cdf.set_ylabel("Fraction of muons above threshold")
    ax_cdf.set_title("Cumulative KE distribution at Al plate\n"
                      "(fraction of arriving muons with KE > threshold)")
    ax_cdf.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, which="both", alpha=0.2)

    # Annotate fraction below 20 MeV
    frac_below_20 = (al["ke_MeV"] < 20).mean()
    ax_cdf.text(0.03, 0.08,
                f"{frac_below_20*100:.1f}% of arriving muons\nhave KE < 20 MeV",
                transform=ax_cdf.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="0.7"))

    fig.suptitle("Muon kinetic energy at Al plate entry\n"
                 "(from entry_ke.csv, volIdx = 4)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "al_entry_ke", fmt)


def fig_decay_secondary_entry_ke(dec_sec_ke, out, fmt):
    """
    Entry KE spectra for Al-plate decay secondaries crossing into each volume.

    Layout: one panel per volume that received any secondary, up to 9 panels
    (3×3 grid).  Each panel shows the KE spectrum broken down by particle
    species (e−, e+, γ), since these are the dominant Michel decay products.

    Skips gracefully if no data is available.
    """
    if dec_sec_ke.empty:
        print("  skipping decay secondary entry KE: "
              "decay_secondary_entry_ke.csv not available or empty.")
        return

    vols_present = [v for v in VOLUME_ORDER if v in dec_sec_ke["volume"].values]
    # Also include volumes not in VOLUME_ORDER (World etc.)
    extra = [v for v in dec_sec_ke["volume"].unique() if v not in vols_present]
    vols_present = vols_present + extra

    if not vols_present:
        print("  skipping decay secondary entry KE: no volume data.")
        return

    n = len(vols_present)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    axes_flat = axes.ravel()

    species = [
        (11,   "e−",    PART_COLOURS["e-"]),
        (-11,  "e+",    PART_COLOURS["e+"]),
        (22,   "γ",     PART_COLOURS["gamma"]),
    ]

    for i, vol in enumerate(vols_present):
        ax  = axes_flat[i]
        sub = dec_sec_ke[dec_sec_ke["volume"] == vol]
        if sub.empty:
            ax.set_visible(False)
            continue

        all_ke = sub["ke_MeV"]
        ke_min = max(all_ke.min(), 0.01)
        ke_max = all_ke.max()
        if ke_max <= ke_min:
            ax.text(0.5, 0.5, "Single KE value", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(VOL_LABELS.get(vol, vol))
            continue

        bins = np.logspace(np.log10(ke_min), np.log10(ke_max), 50)

        # Combined filled
        ax.hist(all_ke, bins=bins,
                color=VOL_COLOURS.get(vol, "#aaa"), alpha=0.3,
                zorder=2, label=f"all  (N={len(all_ke):,})")

        for pdg_val, lbl, col in species:
            s = sub[sub["pdg"] == pdg_val]["ke_MeV"]
            if s.empty: continue
            ax.hist(s, bins=bins, histtype="step", lw=1.4,
                    color=col, zorder=3,
                    label=f"{lbl}  (N={len(s):,})")

        # Median line
        med = all_ke.median()
        ax.axvline(med, color="0.2", ls="--", lw=1.0,
                   label=f"median {med:.1f} MeV")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Entry KE  [MeV]")
        ax.set_ylabel("Secondary entries")
        ax.set_title(VOL_LABELS.get(vol, vol))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.3g}"))
        ax.legend(fontsize=7.5, framealpha=0.7)
        ax.grid(True, which="both", alpha=0.2)

    for i in range(len(vols_present), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "Decay secondary entry KE by volume  —  decay_secondary_entry_ke.csv\n"
        "Secondaries from muon decays in Al plate; KE at each volume boundary crossing",
        fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out, "decay_secondary_entry_ke", fmt)


def fig_stopping(stop, stop_cond, out, fmt):
    if stop.empty and stop_cond.empty:
        return

    vols  = ["Scint_0", "Scint_1", "Al_plate", "Scint_2", "Scint_3",
             "Cu_walls", "Air_hole", "Ceiling", "Roof"]
    parts = ["mu-", "mu+", "e-", "e+"]

    have_local = not stop.empty
    have_cond  = not stop_cond.empty
    vols_local = [v for v in vols if have_local and v in stop["volume"].values]
    vols_cond  = [v for v in vols if have_cond
                  and v != "Scint_0"
                  and v in stop_cond["volume"].values]

    ncols = int(have_local) + int(have_cond)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]
    ax_idx = 0

    if have_local:
        ax = axes[ax_idx]; ax_idx += 1
        x = np.arange(len(vols_local))
        width = 0.8 / len(parts)
        for ip, part in enumerate(parts):
            vals = []
            for v in vols_local:
                row = stop[(stop["volume"] == v) & (stop["particle"] == part)]
                vals.append(row["stop_frac"].values[0] if len(row) else 0.)
            offset = (ip - len(parts) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width, label=part,
                   color=PART_COLOURS[part], edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols_local],
                           rotation=35, ha="right")
        ax.set_title("Local stopping fraction\n(stopped in vol / entered vol)")
        ax.set_ylabel("Fraction")
        ax.grid(axis="y", zorder=0)
        ax.legend(loc="upper right")
        ax.set_yscale("log")

    if have_cond:
        ax = axes[ax_idx]; ax_idx += 1
        x = np.arange(len(vols_cond))
        width = 0.8 / len(parts)
        for ip, part in enumerate(parts):
            vals = []
            for v in vols_cond:
                row = stop_cond[(stop_cond["volume"] == v) &
                                (stop_cond["particle"] == part)]
                vals.append(row["cond_stop_frac"].values[0] if len(row) else 0.)
            offset = (ip - len(parts) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width, label=part,
                   color=PART_COLOURS[part], edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols_cond],
                           rotation=35, ha="right")
        ax.set_title("Conditional stopping fraction\n"
                     "(stopped in vol / entered Scint_0)")
        ax.set_ylabel("Fraction")
        ax.grid(axis="y", zorder=0)
        ax.legend(loc="upper right")
        ax.set_yscale("log")

    ymins, ymaxs = [], []
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        if ymin > 0: ymins.append(ymin)
        if ymax > 0: ymaxs.append(ymax)
    if ymins and ymaxs:
        ymin = min(ymins) * 0.8
        ymax = max(ymaxs) * 1.2
        for ax in axes:
            ax.set_ylim(ymin, ymax)
        print(f"[stopping] y-range = ({ymin:.2e}, {ymax:.2e})")

    fig.suptitle("Particle stopping fractions across the detector stack",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "stopping_fractions", fmt)


def fig_population_flow(stop, out, fmt):
    if stop.empty: return
    vols  = ["Scint_0", "Scint_1", "Al_plate", "Scint_2", "Scint_3"]
    vols  = [v for v in vols if v in stop["volume"].values]
    parts = ["mu-", "mu+", "e-", "e+", "gamma"]
    parts = [p for p in parts if p in stop["particle"].values]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(vols))
    width = 0.8 / len(parts)
    for ip, part in enumerate(parts):
        vals = []
        for v in vols:
            row = stop[(stop["volume"] == v) & (stop["particle"] == part)]
            vals.append(int(row["n_entered"].values[0]) if len(row) else 0)
        offset = (ip - len(parts) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=part,
                      color=PART_COLOURS[part], edgecolor="white", linewidth=0.4, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols], rotation=30, ha="right")
    ax.set_ylabel("Unique tracks entering component")
    ax.set_title("Particle population through the detector stack")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    _save(fig, out, "population_flow", fmt)


def fig_fluence(flu, out, fmt):
    if flu.empty: return
    vols  = ["Scint_0", "Scint_1", "Al_plate", "Scint_2", "Scint_3", "Cu_walls"]
    vols  = [v for v in vols if v in flu["volume"].values]
    parts = ["mu-", "mu+", "e-", "e+", "gamma"]
    parts = [p for p in parts if p in flu["particle"].values]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(vols))
    width = 0.8 / len(parts)
    for ip, part in enumerate(parts):
        vals = []
        for v in vols:
            row = flu[(flu["volume"] == v) & (flu["particle"] == part)]
            vals.append(row["fluence_per_event_per_mm2"].values[0] if len(row) else 0.)
        offset = (ip - len(parts) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=part,
               color=PART_COLOURS[part], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([VOL_LABELS.get(v, v) for v in vols], rotation=30, ha="right")
    ax.set_ylabel(r"Fluence per event  [mm$^{-2}$]")
    ax.set_title("Mean fluence by component and particle species")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    _save(fig, out, "fluence", fmt)


def fig_dedx_profiles(steps, out, fmt):
    if steps.empty: return
    plate_centres = {0: 74, 1: 54, 2: -56, 3: -76}
    half_thick = 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.ravel()
    for idx in range(4):
        ax = axes[idx]
        sub = steps[steps["plateIdx"] == idx]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(f"Scint {idx}")
            continue
        xc = plate_centres[idx]
        xmin, xmax = xc - half_thick, xc + half_thick
        nbins = 40
        bin_edges = np.linspace(xmin, xmax, nbins + 1)
        bin_width  = (xmax - xmin) / nbins
        counts_edep, _ = np.histogram(sub["x"], bins=bin_edges, weights=sub["edep"])
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_events = sub["eventID"].nunique()
        dedx = counts_edep / (bin_width * n_events)
        ax.step(bin_centres, dedx, where="mid", color="0.2", lw=1.3, label="all", zorder=4)
        for part, pdgs in [("mu-", [13]), ("mu+", [-13]),
                           ("e-",  [11]), ("e+",  [-11])]:
            psub = sub[sub["pdg"].isin(pdgs)]
            if psub.empty: continue
            edep_p, _ = np.histogram(psub["x"], bins=bin_edges, weights=psub["edep"])
            dedx_p = edep_p / (bin_width * n_events)
            ax.step(bin_centres, dedx_p, where="mid", lw=1,
                    color=PART_COLOURS[part], label=part, alpha=0.85, zorder=3)
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("dE/dx  [MeV / mm / event]")
        ax.set_title(f"Scint {idx}   (x ∈ [{xmin}, {xmax}] mm)")
        ax.legend(fontsize=8)
        ax.grid(zorder=0)
    fig.suptitle("Depth profiles — mean energy deposition rate", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "dedx_profiles", fmt)


def fig_hit_positions(steps, out, fmt):
    if steps.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for idx in range(4):
        ax = axes[idx]
        sub = steps[steps["plateIdx"] == idx]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(f"Scint {idx}")
            continue
        h = ax.hist2d(sub["y"], sub["z"], bins=40, cmap="inferno",
                      cmin=1, norm=mcolors.LogNorm())
        fig.colorbar(h[3], ax=ax, label="Hits (log scale)")
        ax.set_xlabel("y  [mm]")
        ax.set_ylabel("z  [mm]")
        ax.set_title(f"Scint {idx} — hit map (y–z projection)")
        ax.set_aspect("equal")
    fig.suptitle("Spatial hit distributions on scintillator faces", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "hit_positions", fmt)


def fig_ke_spectra(steps, out, fmt):
    """Step-level KE spectra — all species, from steps.csv (capped).
    Y-axis is dN/dE (counts per MeV) so spectra at different bin densities
    are directly comparable."""
    if steps.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for idx in range(4):
        ax = axes[idx]
        sub = steps[steps["plateIdx"] == idx]
        if sub.empty:
            ax.set_title(f"Scint {idx}")
            continue
        for part, pdgs in [("mu-",  [13]), ("mu+",  [-13]),
                            ("e-",   [11]), ("e+",   [-11])]:
            psub = sub[sub["pdg"].isin(pdgs)]
            ke   = psub["KE"]
            ke   = ke[ke > 0]
            if ke.empty: continue
            bins   = np.logspace(np.log10(max(ke.min(), 1e-3)),
                                 np.log10(ke.max()), 50)
            counts, edges = np.histogram(ke, bins=bins)
            widths  = np.diff(edges)
            centres = 0.5 * (edges[:-1] + edges[1:])
            # Only plot bins with at least one count
            mask = counts > 0
            ax.step(centres[mask], (counts / widths)[mask], where="mid",
                    lw=1.2, label=part, color=PART_COLOURS[part], zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Kinetic energy  [MeV]")
        ax.set_ylabel("dN/dE  [steps / MeV]")
        ax.set_title(f"Scint {idx}  (steps.csv, capped)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", zorder=0)
    fig.suptitle("Step-level KE spectra by particle species\n"
                 "(note: use entry_ke_distributions for muon entry KE with full statistics)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out, "ke_spectra", fmt)


def fig_process_breakdown(steps, out, fmt):
    if steps.empty: return
    sec = steps[steps["parentID"] > 0].copy()
    if sec.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    cmap = plt.cm.Set2
    for idx in range(4):
        ax = axes[idx]
        sub = sec[sec["plateIdx"] == idx]
        if sub.empty:
            ax.set_title(f"Scint {idx}")
            continue
        counts  = sub["process"].value_counts().head(10)
        colours = cmap(np.linspace(0, 1, len(counts)))
        bars = ax.barh(range(len(counts)), counts.values, color=colours,
                       edgecolor="0.3", linewidth=0.4, zorder=3)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Secondary steps")
        ax.set_title(f"Scint {idx}")
        ax.grid(axis="x", zorder=0)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val}", va="center", fontsize=7)
    fig.suptitle("Top creator processes for secondaries in scintillators",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "process_breakdown", fmt)


def fig_detector_schematic(out, fmt):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    components = [
        (+74, 10, "Scint 0", "#66c2a5"),
        (+54, 10, "Scint 1", "#a6d854"),
        (+4,  5,  "Al plate", "#8da0cb"),
        (-56, 10, "Scint 2", "#66c2a5"),
        (-76, 10, "Scint 3", "#a6d854"),
    ]
    y_lo, y_hi = -0.5, 0.5
    ax.add_patch(FancyBboxPatch((-95, y_lo - 0.15), 190, (y_hi - y_lo) + 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor="#f5dde0", edgecolor="#d62728",
                                linewidth=1.5, zorder=1))
    ax.text(95, y_hi + 0.05, "Cu enclosure", fontsize=8,
            ha="right", color="#d62728", style="italic")
    for xc, hx, label, col in components:
        rect = plt.Rectangle((xc - hx, y_lo), 2 * hx, y_hi - y_lo,
                              facecolor=col, edgecolor="0.2",
                              linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(xc, 0, label, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=3)
        ax.text(xc, y_lo - 0.12, f"x = {xc} mm",
                ha="center", fontsize=6.5, color="0.4")
    ax.annotate("", xy=(-90, y_hi + 0.3), xytext=(90, y_hi + 0.3),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"))
    ax.text(0, y_hi + 0.38, r"cosmic $\mu$ direction  (−x)",
            ha="center", fontsize=9, color="#d62728")
    ax.set_xlim(-110, 110)
    ax.set_ylim(y_lo - 0.35, y_hi + 0.55)
    ax.set_xlabel("x  [mm]")
    ax.set_yticks([])
    ax.set_title("Detector cross-section (not to scale in y)")
    ax.set_aspect("auto")
    fig.tight_layout()
    _save(fig, out, "detector_schematic", fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory plot
# ──────────────────────────────────────────────────────────────────────────────

_COMPONENTS = [
    (+74,  10,  314, 500,  "Scint 0",  "#d4edda", "#28a745"),
    (+54,  10,  314, 500,  "Scint 1",  "#d4edda", "#28a745"),
    (-56,  10,  314, 500,  "Scint 2",  "#d4edda", "#28a745"),
    (-76,  10,  314, 500,  "Scint 3",  "#d4edda", "#28a745"),
    (+4,   5,   314, 500,  "Al plate", "#cce5ff", "#0069d9"),
    (0,    95,  315, 360,  "Cu encl.", "#fdf2f2", "#c0392b"),
    (+2685, 100, 1500, 1500, "Ceiling", "#e0e0e0", "#757575"),
    (+5985, 200, 1500, 1500, "Roof",    "#bdbdbd", "#616161"),
]


def _draw_detector_bg(ax, projection="xz", show_labels=True, full_geometry=False):
    for xc, hx, hy, hz, lbl, fc, ec in _COMPONENTS:
        if not full_geometry and lbl in ("Ceiling", "Roof"): continue
        ht = hz if projection == "xz" else hy
        is_cu = lbl.startswith("Cu")
        ax.add_patch(plt.Rectangle(
            (xc - hx, -ht), 2 * hx, 2 * ht,
            facecolor=fc, edgecolor=ec,
            linewidth=1.2 if is_cu else 0.8,
            linestyle="--" if is_cu else "-",
            zorder=0 if is_cu else 1, alpha=0.45))
        if show_labels:
            is_scint = lbl.startswith("Scint")
            ax.text(xc, ht - 0.05 * (2 * ht), lbl,
                    ha="center", va="top", fontsize=7, color=ec,
                    fontweight="bold" if is_scint else "normal",
                    fontstyle="normal" if is_scint else "italic", zorder=5)


def fig_trajectories(steps, traj, out, fmt, n_events=6, projection="xz"):
    if not traj.empty:
        data   = traj
        source = "trajectories.csv"
    elif not steps.empty:
        data   = steps
        source = "steps.csv"
    else:
        return
    print(f"  Trajectory source: {source}")
    t_col   = "z" if projection == "xz" else "y"
    t_label = "z" if projection == "xz" else "y"

    if "plateIdx" in data.columns:
        plate_counts = data.groupby("eventID")["plateIdx"].nunique()
        good_events  = plate_counts[plate_counts >= 3].index.tolist()
        if not good_events:
            good_events = sorted(data["eventID"].unique().tolist())
    else:
        good_events = sorted(data["eventID"].unique().tolist())

    if len(good_events) > n_events:
        indices  = np.linspace(0, len(good_events) - 1, n_events, dtype=int)
        selected = [good_events[i] for i in indices]
    else:
        selected = good_events[:n_events]
    if not selected: return

    x_global_min = data["x"].min()
    x_global_max = data["x"].max()
    full_geom    = (x_global_max - x_global_min) > 500

    n = len(selected)
    if   n <= 2: ncols, nrows = n,    1
    elif n <= 4: ncols, nrows = 2,    2
    elif n <= 6: ncols, nrows = 3,    2
    else:        ncols, nrows = 3, (n + 2) // 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    def _style(pdg):
        if pdg ==  13: return PART_COLOURS["mu-"],   2.0, 1.0, 10
        if pdg == -13: return PART_COLOURS["mu+"],   2.0, 1.0, 10
        if pdg ==  11: return PART_COLOURS["e-"],    0.9, 0.7,  8
        if pdg == -11: return PART_COLOURS["e+"],    0.9, 0.7,  8
        if pdg ==  22: return PART_COLOURS["gamma"], 0.7, 0.5,  6
        return "#8c564b", 0.6, 0.4, 5

    for idx, evtID in enumerate(selected):
        ax  = axes_flat[idx]
        evt = data[data["eventID"] == evtID].copy()
        evt = evt.sort_values(["trackID", "x"], ascending=[True, False])
        x_min_evt = evt["x"].min()
        x_max_evt = evt["x"].max()
        x_pad     = max((x_max_evt - x_min_evt) * 0.05, 100)
        t_vals    = evt[t_col]
        t_data_min, t_data_max = t_vals.min(), t_vals.max()
        t_hw_idx  = 3 if projection == "xz" else 2
        max_hw    = 0
        for comp in _COMPONENTS:
            lbl = comp[4]
            if not full_geom and lbl in ("Ceiling", "Roof"): continue
            max_hw = max(max_hw, comp[t_hw_idx])
        t_lo = min(t_data_min, -max_hw)
        t_hi = max(t_data_max,  max_hw)
        _draw_detector_bg(ax, projection=projection,
                          show_labels=(idx == 0), full_geometry=full_geom)
        for trkID, grp in evt.groupby("trackID"):
            grp  = grp.sort_values("x", ascending=False)
            xs   = grp["x"].values
            ts   = grp[t_col].values
            pdg  = grp["pdg"].iloc[0]
            edeps = grp["edep"].values if "edep" in grp.columns else np.zeros(len(grp))
            col, lw, alpha, zo = _style(pdg)
            ax.plot(xs, ts, color=col, linewidth=lw, alpha=alpha,
                    zorder=zo, solid_capstyle="round")
            sizes = np.clip(edeps * 8, 1, 40)
            ax.scatter(xs, ts, s=sizes, color=col, alpha=alpha * 0.8,
                       edgecolors="none", zorder=zo + 1)
            if grp["parentID"].iloc[0] > 0:
                parent_steps = evt[evt["trackID"] == grp["parentID"].iloc[0]]
                if not parent_steps.empty:
                    p_xs    = parent_steps["x"].values
                    p_ts    = parent_steps[t_col].values
                    dists   = np.abs(p_xs - xs[0])
                    nearest = np.argmin(dists)
                    ax.plot([p_xs[nearest], xs[0]], [p_ts[nearest], ts[0]],
                            color=col, linewidth=0.5, alpha=0.4,
                            linestyle=":", zorder=zo - 1)
        ax.set_xlim(x_max_evt + x_pad, min(x_min_evt - x_pad, -110))
        ax.set_ylim(t_lo, t_hi)
        ax.set_xlabel("x  [mm]", fontsize=9)
        ax.set_ylabel(f"{t_label}  [mm]", fontsize=9)
        ax.set_title(f"Event {int(evtID)}", fontsize=10)
        ax.tick_params(labelsize=8)

    for idx in range(len(selected), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PART_COLOURS["mu-"],   lw=2,   label="μ⁻"),
        Line2D([0], [0], color=PART_COLOURS["mu+"],   lw=2,   label="μ⁺"),
        Line2D([0], [0], color=PART_COLOURS["e-"],    lw=1,   label="e⁻"),
        Line2D([0], [0], color=PART_COLOURS["e+"],    lw=1,   label="e⁺"),
        Line2D([0], [0], color=PART_COLOURS["gamma"], lw=0.8, label="γ"),
    ]
    axes_flat[0].legend(handles=legend_elements, loc="lower left",
                        fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Particle trajectories through full geometry  "
        f"(x–{t_label} projection, {len(selected)} events)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out, "trajectories", fmt)


# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir, name, fmt):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.{fmt}")


# ──────────────────────────────────────────────────────────────────────────────
# Decay loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_decay(d):
    """
    Load decay.csv — one row per muon decay step detected in any volume.
    Columns: eventID, volume, pdg, trackID, ke_MeV, x_mm, y_mm, z_mm, time_ns
    """
    p = d / "decay.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())),
                     on_bad_lines="skip")
    for col in ["pdg", "trackID", "ke_MeV", "x_mm", "y_mm", "z_mm", "time_ns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["volume", "ke_MeV"])
    n_al = (df["volume"] == "Al_plate").sum()
    print(f"  decay: {len(df):,} total decay records  "
          f"({n_al:,} in Al plate)")
    return df


def load_decay_summary(d):
    """
    Load decay_summary.csv — per-volume decay counts from RunAction.
    Columns: volume, n_decay_mum, n_decay_mup, n_decay_total,
             decay_rate_mum_per_event, decay_rate_mup_per_event
    """
    p = d / "decay_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    return df


def load_decay_secondary_stopping(d):
    """
    Load decay_secondary_stopping.csv — per-volume stopping counts for
    secondaries produced by muon decays in the Al plate.
    Columns: volume, particle, n_entered, n_stopped, stop_frac,
             stop_frac_vs_total
    Only rows where n_entered > 0 are written by RunAction.
    """
    p = d / "decay_secondary_stopping.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())))
    for col in ["n_entered", "n_stopped", "stop_frac", "stop_frac_vs_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["volume", "particle", "n_entered"])
    if df.empty:
        return pd.DataFrame()
    total = int(df[df["particle"] == "all"]["n_entered"].sum()) if "all" in df["particle"].values else 0
    print(f"  decay_secondary_stopping: {len(df):,} rows  "
          f"({total:,} total secondary volume-entries)")
    return df


def load_coincidences(d):
    """
    Load coincidences.csv — one row per event where ≥1 scintillator fired.
    Columns: eventID, coinc_mask, n_fired, al_fired,
             edep_s0_MeV…edep_s3_MeV, edep_al_MeV
    coinc_mask is a 4-bit integer: bit i = Scint_i above threshold.
    """
    p = d / "coincidences.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(pd.io.common.StringIO(_clean(p.read_text())),
                     on_bad_lines="skip")
    for col in ["coinc_mask", "n_fired", "al_fired",
                "edep_s0_MeV", "edep_s1_MeV", "edep_s2_MeV",
                "edep_s3_MeV", "edep_al_MeV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["coinc_mask", "n_fired"])
    df["coinc_mask"] = df["coinc_mask"].astype(int)
    df["n_fired"]    = df["n_fired"].astype(int)
    n4 = (df["n_fired"] == 4).sum()
    print(f"  coincidences: {len(df):,} triggered events  ({n4:,} 4-fold)")
    return df


def load_coincidence_summary(d):
    """
    Load coincidence_summary.csv — run-level coincidence counts and plate
    efficiencies.  The file has comment lines (starting with '#') and several
    labelled sections.  Returns a dict keyed by section name:
      'classes', 'pairs', 'efficiency', 'edep', 'patterns'
    """
    import io as _io
    p = d / "coincidence_summary.csv"
    if not p.exists():
        return {}
    text  = _clean(p.read_text())
    lines = text.splitlines()

    header_map = {
        "class,n_events,rate_per_event":                     "classes",
        "pair,n_events,rate_per_event":                      "pairs",
        "plate,n_trigger,n_4fold,efficiency,efficiency_err": "efficiency",
        "plate,n_coinc_events,mean_edep_MeV":                "edep",
        "mask,label,n_events,rate_per_event,n_with_al":      "patterns",
    }

    sections: dict = {}
    current = None
    buf: list = []

    def _flush():
        if current and buf:
            try:
                sections[current] = pd.read_csv(_io.StringIO("\n".join(buf)))
            except Exception:
                pass

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in header_map:
            _flush()
            current = header_map[s]
            buf = [s]
        elif current:
            buf.append(s)
    _flush()

    if sections:
        print(f"  coincidence_summary: sections {list(sections.keys())}")
    return sections


# ──────────────────────────────────────────────────────────────────────────────
# Decay figures
# ──────────────────────────────────────────────────────────────────────────────

# Colour scheme consistent with the rest of the plots
_MUM_COL = PART_COLOURS["mu-"]
_MUP_COL = PART_COLOURS["mu+"]
_AL_COL  = VOL_COLOURS["Al_plate"]


def fig_decay_overview(decay, decay_sum, out, fmt):
    """
    Two-panel overview figure.

    Left  — bar chart of total decay counts per volume (μ− and μ+ stacked),
            with per-event rates annotated.  Gives immediate answer to
            "where do muons decay in this detector?"

    Right — same data as a table with absolute counts and per-event rates,
            useful for quick numerical reference.

    If decay counts are zero or very low (expected for this energy range)
    the figure still renders and notes the low statistics clearly.
    """
    if decay_sum.empty:
        print("  skipping decay overview: decay_summary.csv not available.")
        return

    # Remove TOTAL row for the bar chart
    plot_df = decay_sum[decay_sum["volume"] != "TOTAL"].copy()
    if plot_df.empty:
        print("  skipping decay overview: no volume rows in decay_summary.csv.")
        return

    # Determine total events from TOTAL row if present
    total_row = decay_sum[decay_sum["volume"] == "TOTAL"]
    n_total_decays = int(total_row["n_decay_total"].values[0]) \
                     if not total_row.empty else int(plot_df["n_decay_total"].sum())

    fig, (ax_bar, ax_tab) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1.6, 1]})

    # ── Left: stacked bar chart ──────────────────────────────────────────────
    vols = plot_df["volume"].tolist()
    x    = np.arange(len(vols))
    nm   = plot_df["n_decay_mum"].values.astype(float)
    np_  = plot_df["n_decay_mup"].values.astype(float)

    bars_m = ax_bar.bar(x, nm, color=_MUM_COL, edgecolor="white",
                         linewidth=0.4, label="μ−", zorder=3)
    bars_p = ax_bar.bar(x, np_, bottom=nm, color=_MUP_COL, edgecolor="white",
                         linewidth=0.4, label="μ+", zorder=3)

    # Annotate total count above each bar
    for xi, (n_m, n_p) in enumerate(zip(nm, np_)):
        total = n_m + n_p
        if total > 0:
            ax_bar.text(xi, total + 0.3, f"{int(total)}",
                        ha="center", va="bottom", fontsize=8)

    # Highlight Al plate bar
    al_idx = vols.index("Al_plate") if "Al_plate" in vols else -1
    if al_idx >= 0:
        ax_bar.get_children()[al_idx].set_edgecolor("#1a1a1a")
        ax_bar.get_children()[al_idx].set_linewidth(1.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([VOL_LABELS.get(v, v) for v in vols],
                            rotation=35, ha="right")
    ax_bar.set_ylabel("Muon decays (full run)")
    ax_bar.set_title("Muon decays per detector component\n"
                      "(Al plate highlighted)")
    ax_bar.legend(loc="upper right")
    ax_bar.grid(axis="y", zorder=0)
    if ax_bar.get_ylim()[1] > 0:
        ax_bar.set_yscale("symlog", linthresh=1)

    if n_total_decays == 0:
        ax_bar.text(0.5, 0.6,
                    "No decays recorded\n"
                    "(expected: muon range >> plate thickness\n"
                    " for E > 100 MeV; see decay_summary.csv)",
                    transform=ax_bar.transAxes,
                    ha="center", va="center", fontsize=10,
                    color="0.5",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    # ── Right: summary table ─────────────────────────────────────────────────
    ax_tab.axis("off")
    col_labels = ["Volume", "μ− decays", "μ+ decays", "Total",
                  "rate/evt\n(μ−)", "rate/evt\n(μ+)"]
    # Fill NaN with 0 before formatting (can appear in merged runs with
    # no decays in a given volume)
    decay_sum_clean = decay_sum.fillna(0)
    table_data = []
    for _, row in decay_sum_clean.iterrows():
        table_data.append([
            VOL_LABELS.get(row["volume"], row["volume"]),
            f"{int(row['n_decay_mum']):,}",
            f"{int(row['n_decay_mup']):,}",
            f"{int(row['n_decay_total']):,}",
            f"{row['decay_rate_mum_per_event']:.2e}",
            f"{row['decay_rate_mup_per_event']:.2e}",
        ])

    tbl = ax_tab.table(cellText=table_data,
                        colLabels=col_labels,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)

    # Bold the Al plate row and TOTAL row
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        text = cell.get_text().get_text()
        if "Al plate" in text or "TOTAL" in text:
            cell.set_facecolor("#fff3cd")
            cell.set_text_props(fontweight="bold")

    ax_tab.set_title("Decay counts summary", fontsize=11, pad=12)

    fig.suptitle("Muon decay summary across all detector volumes",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "decay_overview", fmt)


def fig_decay_detail(decay, out, fmt):
    """
    Four-panel detailed figure using raw decay.csv records.

    Top-left    — KE at decay point: μ− and μ+ histograms per volume.
                  Low KE confirms muons stopped before decaying.
    Top-right   — Decay positions: x distribution showing which depth
                  in the detector the decay occurred.
    Bottom-left — Decay time distribution (global time since event start).
                  Muons that stop and decay show the ~2.2 μs lifetime.
    Bottom-right— 2D scatter of decay position (x vs y) coloured by
                  volume, giving spatial context.

    Skips gracefully if no decay records are available.
    """
    if decay.empty:
        print("  skipping decay detail: no records in decay.csv.")
        return

    # Warn clearly if statistics are very low
    n = len(decay)
    low_stats = n < 10

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax_ke, ax_x), (ax_t, ax_xy) = axes

    vols_present = decay["volume"].unique()
    vol_colours  = {v: VOL_COLOURS.get(v, "#888") for v in vols_present}

    # ── Top-left: KE at decay ────────────────────────────────────────────────
    for label, pdg_val, col in [("μ−", 13, _MUM_COL), ("μ+", -13, _MUP_COL)]:
        sub = decay[decay["pdg"] == pdg_val]["ke_MeV"]
        if sub.empty:
            continue
        if sub.max() > sub.min():
            bins = np.linspace(0, max(sub.max() * 1.05, 1.0), 40)
        else:
            bins = np.linspace(0, 10, 40)
        ax_ke.hist(sub, bins=bins, histtype="step", lw=1.8,
                   color=col, label=f"{label}  (N={len(sub):,})")
    ax_ke.set_xlabel("Muon KE at decay step  [MeV]")
    ax_ke.set_ylabel("Decay events")
    ax_ke.set_title("Kinetic energy at decay point\n"
                    "(near zero confirms muon stopped before decaying)")
    ax_ke.legend(fontsize=9)
    ax_ke.grid(axis="y", alpha=0.3)
    if low_stats:
        ax_ke.text(0.5, 0.5, f"Low statistics  (N={n})",
                   transform=ax_ke.transAxes, ha="center", va="center",
                   color="0.5", fontsize=11)

    # ── Top-right: x position distribution ──────────────────────────────────
    for vol in vols_present:
        sub = decay[decay["volume"] == vol]["x_mm"]
        if sub.empty:
            continue
        ax_x.hist(sub, bins=30, histtype="step", lw=1.5,
                  color=vol_colours[vol],
                  label=f"{VOL_LABELS.get(vol, vol)}  (N={len(sub):,})")
    ax_x.set_xlabel("x  [mm]  (muon beam direction)")
    ax_x.set_ylabel("Decay events")
    ax_x.set_title("Decay position along beam axis\n"
                    "(x coordinate inside detector)")
    ax_x.legend(fontsize=8)
    ax_x.grid(axis="y", alpha=0.3)

    # ── Bottom-left: decay time ──────────────────────────────────────────────
    t = decay["time_ns"]
    if not t.empty and t.max() > 0:
        t_max  = min(t.max(), 5000.0)   # cap at 5 μs for readability
        bins_t = np.linspace(0, t_max, 50)
        ax_t.hist(t[t <= t_max], bins=bins_t,
                  color=_AL_COL, edgecolor="0.3", linewidth=0.4,
                  alpha=0.8, zorder=3, label="all volumes")

        # Overlay muon lifetime reference: dN/dt ∝ exp(-t/τ), τ_μ = 2197 ns
        tau_ns = 2197.0
        t_ref  = np.linspace(0, t_max, 300)
        norm   = len(decay) * (bins_t[1] - bins_t[0]) / tau_ns
        ax_t.plot(t_ref, norm * np.exp(-t_ref / tau_ns),
                  "k--", lw=1.4, alpha=0.7,
                  label=r"$e^{-t/\tau_\mu}$,  $\tau_\mu$ = 2197 ns")
        ax_t.set_yscale("log")
    else:
        ax_t.text(0.5, 0.5, "No time data available",
                  transform=ax_t.transAxes, ha="center", va="center",
                  color="0.5", fontsize=11)

    ax_t.set_xlabel("Global time at decay  [ns]")
    ax_t.set_ylabel("Decay events (log scale)")
    ax_t.set_title("Decay time distribution\n"
                    r"(exponential with $\tau_\mu \approx 2197$ ns for stopped muons)")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, which="both", alpha=0.2)

    # ── Bottom-right: 2D position scatter ───────────────────────────────────
    for vol in vols_present:
        sub = decay[decay["volume"] == vol]
        if sub.empty:
            continue
        ax_xy.scatter(sub["x_mm"], sub["y_mm"],
                      s=20, alpha=0.7, edgecolors="none",
                      color=vol_colours[vol],
                      label=f"{VOL_LABELS.get(vol, vol)}")
    ax_xy.set_xlabel("x  [mm]")
    ax_xy.set_ylabel("y  [mm]")
    ax_xy.set_title("Decay positions  (x–y projection)")
    ax_xy.legend(fontsize=8, markerscale=1.5)
    ax_xy.grid(alpha=0.2)
    if low_stats:
        ax_xy.text(0.05, 0.95, f"N = {n} total decays",
                   transform=ax_xy.transAxes, va="top", fontsize=9,
                   color="0.4")

    fig.suptitle(
        "Muon decay detail  —  decay.csv\n"
        "(all volumes; filter to Al_plate for absorber-specific analysis)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "decay_detail", fmt)


def fig_decay_al_plate(decay, out, fmt):
    """
    Al-plate-specific decay figure.  Only rendered if at least one decay
    was recorded in the Al plate; otherwise prints a clear message.

    Three panels:
      Left   — KE at decay for μ− and μ+ in Al only.
      Centre — y–z position of decays in the Al plate face.
      Right  — Decay time with muon lifetime reference.
    """
    if decay.empty:
        print("  skipping Al plate decay plot: no decay records.")
        return

    al = decay[decay["volume"] == "Al_plate"].copy()
    if al.empty:
        print("  skipping Al plate decay plot: zero decays recorded in Al plate.\n"
              "  (Expected — muon range >> 10 mm Al for E > 100 MeV.\n"
              "   Decays may appear after longer runs or if KE range is extended.)")
        return

    print(f"  Al plate decays: {len(al):,}  "
          f"(μ− {(al['pdg']==13).sum():,},  "
          f"μ+ {(al['pdg']==-13).sum():,})")

    fig, (ax_ke, ax_yz, ax_t) = plt.subplots(1, 3, figsize=(15, 5))

    # ── Left: KE at decay ────────────────────────────────────────────────────
    for label, pdg_val, col in [("μ−", 13, _MUM_COL), ("μ+", -13, _MUP_COL)]:
        sub = al[al["pdg"] == pdg_val]["ke_MeV"]
        if sub.empty:
            continue
        bins = np.linspace(0, max(sub.max() * 1.1, 1.0), 30)
        ax_ke.hist(sub, bins=bins, color=col, alpha=0.6,
                   edgecolor="0.3", linewidth=0.4,
                   label=f"{label}  (N={len(sub):,})")
    ax_ke.set_xlabel("KE at decay  [MeV]")
    ax_ke.set_ylabel("Events")
    ax_ke.set_title("Muon KE at decay in Al plate\n"
                    "(should be ≈ 0 for stopped muons)")
    ax_ke.legend(fontsize=9)
    ax_ke.grid(axis="y", alpha=0.3)

    # ── Centre: y–z position ─────────────────────────────────────────────────
    if len(al) > 1:
        h = ax_yz.hist2d(al["y_mm"], al["z_mm"], bins=20,
                          cmap="plasma", cmin=1)
        fig.colorbar(h[3], ax=ax_yz, label="Decay events")
    else:
        ax_yz.scatter(al["y_mm"], al["z_mm"], s=50,
                       color=_AL_COL, edgecolors="0.2", zorder=3)
    ax_yz.set_xlabel("y  [mm]")
    ax_yz.set_ylabel("z  [mm]")
    ax_yz.set_title("Decay positions in Al plate\n(y–z face projection)")
    ax_yz.set_aspect("equal")
    ax_yz.grid(alpha=0.2)

    # ── Right: decay time ─────────────────────────────────────────────────────
    t = al["time_ns"]
    if t.max() > 0:
        t_max  = min(t.max() * 1.05, 10000.0)
        bins_t = np.linspace(0, t_max, 40)
        ax_t.hist(t, bins=bins_t, color=_AL_COL,
                  edgecolor="0.3", linewidth=0.4, alpha=0.85,
                  label="Al plate decays")
        tau_ns = 2197.0
        t_ref  = np.linspace(0, t_max, 300)
        norm   = len(al) * (bins_t[1] - bins_t[0]) / tau_ns
        ax_t.plot(t_ref, norm * np.exp(-t_ref / tau_ns),
                  "k--", lw=1.4, alpha=0.7,
                  label=r"$e^{-t/\tau_\mu}$  (free $\tau$ = 2197 ns)")
        ax_t.set_yscale("log")

    ax_t.set_xlabel("Global time at decay  [ns]")
    ax_t.set_ylabel("Events (log scale)")
    ax_t.set_title("Decay time in Al plate\n"
                    r"($\tau_{\mu^-}$ in Al ≈ 864 ns due to nuclear capture)")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, which="both", alpha=0.2)

    # Note: μ- in Al has shortened lifetime due to muon nuclear capture
    # (competing with decay). Free muon τ = 2197 ns; in Al τ_eff ≈ 864 ns.
    ax_t.text(0.97, 0.97,
              "Note: μ− lifetime in Al\n≈ 864 ns (nuclear capture\ncompetes with decay)",
              transform=ax_t.transAxes, ha="right", va="top", fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="0.7"))

    fig.suptitle(
        f"Muon decay in Al plate  —  {len(al):,} events\n"
        "(low count expected: muon range >> 10 mm for E > 100 MeV source)",
        fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out, "decay_al_plate", fmt)


def fig_decay_secondary_stopping(dec_sec, out, fmt):
    """
    Three-panel figure showing where Al-plate decay secondaries end up.

    For muon decay (μ± → e± + ν + ν̄), the direct secondaries are:
      e−  (Michel electron from μ−)  or  e+  (positron from μ+)
      νμ / ν̄μ   (muon neutrino/antineutrino)
      νe / ν̄e   (electron neutrino/antineutrino)

    Neutrinos are tracked by Geant4 but do not interact — they pass
    straight through all volumes and are recorded as "entered but never
    stopped" in World, which is correct.  They appear here as "ν (other)"
    since PDGToPartIdx maps all non-μ/e/γ PDG codes to kOther.

    Left   — n_entered per volume per particle species (stacked bar).
    Centre — local stopping fraction (n_stopped / n_entered) per volume.
    Right  — stop_frac_vs_total: absolute probability of ending up stopped
              in each volume.  This is the key physics result.
    """
    if dec_sec.empty:
        print("  skipping decay secondary stopping: "
              "decay_secondary_stopping.csv not available or empty.\n"
              "  (Expected when no Al-plate decays recorded — "
              "run more events or extend energy range.)")
        return

    vols  = [v for v in VOLUME_ORDER if v in dec_sec["volume"].values]
    if not vols:
        print("  skipping decay secondary stopping: no usable rows.")
        return

    # Rename "other" → "ν (neutrino)" for muon decay context,
    # and build the display species list in physically meaningful order
    SPECIES_RENAME = {
        "e-":    "e−  (Michel)",
        "e+":    "e+  (Michel)",
        "gamma": "γ",
        "other": "ν  (neutrino)",
        "all":   "all",
    }
    SPECIES_COLOURS = {
        "e-":    PART_COLOURS["e-"],
        "e+":    PART_COLOURS["e+"],
        "gamma": PART_COLOURS["gamma"],
        "other": "#aaaaaa",
        "all":   PART_COLOURS["all"],
    }

    all_parts_present = dec_sec["particle"].unique().tolist()
    # Order: e-, e+, gamma, other, then all
    ordered = [p for p in ["e-", "e+", "gamma", "other", "all"]
               if p in all_parts_present]

    fig, (ax_ent, ax_loc, ax_tot) = plt.subplots(1, 3, figsize=(18, 5))
    x  = np.arange(len(vols))
    w2 = 0.8 / max(len([p for p in ordered if p != "all"]), 1)

    # ── Left: n_entered stacked bar (exclude "all" to avoid double-count) ────
    plot_parts = [p for p in ordered if p != "all"]
    bottoms = np.zeros(len(vols))
    for p in plot_parts:
        vals = np.array([
            float(dec_sec.loc[(dec_sec["volume"] == v) &
                               (dec_sec["particle"] == p), "n_entered"]
                            .values[0])
            if len(dec_sec.loc[(dec_sec["volume"] == v) &
                                (dec_sec["particle"] == p)]) else 0.
            for v in vols
        ])
        ax_ent.bar(x, vals, bottom=bottoms, width=0.7,
                   color=SPECIES_COLOURS[p], edgecolor="white",
                   linewidth=0.4, label=SPECIES_RENAME[p], zorder=3)
        bottoms += vals

    ax_ent.set_xticks(x)
    ax_ent.set_xticklabels([VOL_LABELS.get(v, v) for v in vols],
                            rotation=35, ha="right")
    ax_ent.set_ylabel("Secondary tracks entering volume")
    ax_ent.set_title("Decay secondary reach per volume\n"
                      "(stacked by species; ν pass through, e± stop)")
    ax_ent.legend(fontsize=8, loc="upper right")
    ax_ent.grid(axis="y", zorder=0)
    if bottoms.max() > 0:
        ax_ent.set_yscale("symlog", linthresh=1)

    # ── Centre: local stop_frac — exclude neutrinos (they never stop) ────────
    show_parts = [p for p in ["e-", "e+", "gamma", "all"] if p in all_parts_present]
    nsp = max(len(show_parts), 1)
    w3  = 0.8 / nsp
    for ip, p in enumerate(show_parts):
        vals = np.array([
            float(dec_sec.loc[(dec_sec["volume"] == v) &
                               (dec_sec["particle"] == p), "stop_frac"]
                            .values[0])
            if len(dec_sec.loc[(dec_sec["volume"] == v) &
                                (dec_sec["particle"] == p)]) else 0.
            for v in vols
        ])
        offset = (ip - nsp / 2 + 0.5) * w3
        ax_loc.bar(x + offset, vals, width=w3,
                   color=SPECIES_COLOURS[p], edgecolor="white",
                   linewidth=0.4, label=SPECIES_RENAME[p], zorder=3)

    ax_loc.set_xticks(x)
    ax_loc.set_xticklabels([VOL_LABELS.get(v, v) for v in vols],
                            rotation=35, ha="right")
    ax_loc.set_ylabel("Local stopping fraction\n(stopped / entered this vol)")
    ax_loc.set_title("Local stopping fraction\n"
                      "(ν omitted — they never stop)")
    ax_loc.legend(fontsize=8, loc="upper right")
    ax_loc.grid(axis="y", zorder=0)
    if ax_loc.get_ylim()[1] > 0:
        ax_loc.set_yscale("log")

    # ── Right: stop_frac_vs_total ─────────────────────────────────────────────
    for ip, p in enumerate(show_parts):
        vals = np.array([
            float(dec_sec.loc[(dec_sec["volume"] == v) &
                               (dec_sec["particle"] == p), "stop_frac_vs_total"]
                            .values[0])
            if len(dec_sec.loc[(dec_sec["volume"] == v) &
                                (dec_sec["particle"] == p)]) else 0.
            for v in vols
        ])
        offset = (ip - nsp / 2 + 0.5) * w3
        ax_tot.bar(x + offset, vals, width=w3,
                   color=SPECIES_COLOURS[p], edgecolor="white",
                   linewidth=0.4, label=SPECIES_RENAME[p], zorder=3)

    ax_tot.set_xticks(x)
    ax_tot.set_xticklabels([VOL_LABELS.get(v, v) for v in vols],
                            rotation=35, ha="right")
    ax_tot.set_ylabel("Absolute stopping fraction\n"
                       "(stopped here / all secondary crossings)")
    ax_tot.set_title("Where decay products stop\n"
                      "(key result: fraction of all secondaries ending in vol X)")
    ax_tot.legend(fontsize=8, loc="upper right")
    ax_tot.grid(axis="y", zorder=0)
    if ax_tot.get_ylim()[1] > 0:
        ax_tot.set_yscale("log")

    # Mark Al plate on all three panels
    if "Al_plate" in vols:
        al_x = vols.index("Al_plate")
        for ax in (ax_ent, ax_loc, ax_tot):
            ax.axvline(al_x, color="#c0392b", lw=1.0, ls="--",
                       alpha=0.5, zorder=1)

    # Physics note
    fig.text(0.5, -0.04,
             "μ− → e− + ν̄e + νμ  |  μ+ → e+ + νe + ν̄μ\n"
             "ν recorded as 'other' (PDG ±12, ±14); they enter volumes but never stop  —"
             "  only e± and γ contribute to detector response",
             ha="center", va="top", fontsize=8, color="0.4",
             style="italic")

    fig.suptitle(
        "Al-plate decay secondary stopping  —  decay_secondary_stopping.csv\n"
        "Secondaries = direct decay products of muons that decayed in the Al plate",
        fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out, "decay_secondary_stopping", fmt)



def fig_energy_loss_cascade(entry_ke, gun, out, fmt):
    """
    Energy landscape through the full detector stack.
 
    Stage order: Gun → Scint 0 → Scint 1 → Al plate → Scint 2 → Scint 3
 
    Two panels:
 
    Top — dN/dE spectra for all six stages overlaid on a common log-log
          axis, coloured by position in the stack.  Shows the progressive
          spectral hardening: low-energy muons are preferentially absorbed
          upstream, shifting the median rightward at each stage.
 
    Bottom — Energy-dependent transmission between consecutive stages:
          T(E) = (dN/dE)_downstream / (dN/dE)_upstream
 
          T = 1 means the downstream stage received the same flux at that
          energy as the upstream stage (no absorption).
          T < 1 means that energy bin was partially absorbed in the
          intervening material.
          The shape of T(E) reveals the material's stopping power as a
          function of muon energy — flat at high E (minimum ionising),
          dropping sharply at low E (range limit).
    """
    if entry_ke.empty and gun.empty:
        print("  skipping energy cascade: no data available.")
        return
 
    muons_ke = entry_ke[entry_ke["pdg"].isin([13, -13])].copy() \
               if not entry_ke.empty else pd.DataFrame()
 
    # ── Build per-stage KE arrays ─────────────────────────────────────────────
    # Stage 0: gun (before any material) — spectrum panel only
    # Stages 1–5: volIdx 0, 1, 4, 2, 3  (physical order through detector)
    stage_order = [
        ("Gun",     None,  "#555555"),
        ("Scint 0", 0,     SCINT_COLOURS[0]),
        ("Scint 1", 1,     SCINT_COLOURS[1]),
        ("Al plate",4,     VOL_COLOURS["Al_plate"]),
        ("Scint 2", 2,     SCINT_COLOURS[2]),
        ("Scint 3", 3,     SCINT_COLOURS[3]),
    ]
 
    ke_arrays = {}
    for label, vol_idx, _ in stage_order:
        if label == "Gun":
            if not gun.empty:
                ke_arrays[label] = gun[gun["pdg"].isin([13, -13])]["energy_MeV"].values
            else:
                ke_arrays[label] = np.array([])
        else:
            if not muons_ke.empty and vol_idx is not None:
                ke_arrays[label] = muons_ke[muons_ke["volIdx"] == vol_idx]["ke_MeV"].values
            else:
                ke_arrays[label] = np.array([])
 
    # Drop stages with no data
    stage_order = [(lbl, vi, col) for lbl, vi, col in stage_order
                   if len(ke_arrays[lbl]) > 0]
    if len(stage_order) < 2:
        print("  skipping energy cascade: need at least 2 stages with data.")
        return
 
    # Detector stages only (exclude Gun) — used for survival fraction
    detector_stages = [(lbl, vi, col) for lbl, vi, col in stage_order
                       if lbl != "Gun"]
 
    # Common bin edges spanning all stages
    all_ke = np.concatenate([ke_arrays[lbl] for lbl, _, _ in stage_order])
    ke_min = max(all_ke.min(), 1.0)
    ke_max = all_ke.max()
    bins    = np.logspace(np.log10(ke_min), np.log10(ke_max), 80)
    widths  = np.diff(bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
 
    # Pre-compute dN/dE for each stage
    dndE = {}
    for label, _, _ in stage_order:
        counts, _ = np.histogram(ke_arrays[label], bins=bins)
        dndE[label] = counts / widths
 
    fig, (ax_spec, ax_ratio) = plt.subplots(2, 1, figsize=(11, 10),
                                              gridspec_kw={"height_ratios": [1.6, 1]})
 
    # ── Top panel: overlaid dN/dE spectra ────────────────────────────────────
    for label, _, col in stage_order:
        d = dndE[label]
        mask = d > 0
        n = len(ke_arrays[label])
        ax_spec.step(centres[mask], d[mask], where="mid",
                     color=col, lw=1.8, label=f"{label}  (N={n:,})")
 
    # Annotate median shift across stages
    for label, _, col in stage_order:
        arr = ke_arrays[label]
        if len(arr) > 10:
            med = np.median(arr)
            ax_spec.axvline(med, color=col, lw=0.7, ls=":", alpha=0.6)
 
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("Kinetic energy  [MeV]")
    ax_spec.set_ylabel("dN/dE  [muons / MeV]")
    ax_spec.set_title("Muon energy spectrum at each stage\n"
                       "(dotted verticals = stage median; "
                       "spectrum hardens downstream)")
    ax_spec.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax_spec.legend(fontsize=9, framealpha=0.9)
    ax_spec.grid(True, which="both", alpha=0.2)
 
    # ── Bottom panel: cumulative survival fraction S(E) ──────────────────────
    # S(E) = N_downstream(>E) / N_upstream(>E)
    #
    # Gun is excluded — it samples all generated muons while detector stages
    # sample only those hitting the detector face.  Geometric acceptance is
    # energy-independent so Gun/Scint0 would just give a flat line at the
    # acceptance fraction, not absorption information.  Starting from Scint 0
    # as the reference, every ratio compares the same intercepting population
    # and S(E) < 1 unambiguously means tracks were absorbed, not missed.
    trans_colours = ["#e67e22", "#27ae60", "#8e44ad", "#2980b9", "#c0392b"]
 
    e_axis = np.logspace(np.log10(ke_min), np.log10(ke_max), 400)
 
    if len(detector_stages) < 2:
        ax_ratio.text(0.5, 0.5, "Need ≥ 2 detector stages with data",
                      ha="center", va="center", transform=ax_ratio.transAxes,
                      color="0.5", fontsize=11)
    else:
        for i in range(len(detector_stages) - 1):
            lbl_up   = detector_stages[i][0]
            lbl_down = detector_stages[i + 1][0]
            col      = trans_colours[i % len(trans_colours)]
 
            arr_up   = ke_arrays[lbl_up]
            arr_down = ke_arrays[lbl_down]
 
            n_up   = np.array([(arr_up   > e).sum() for e in e_axis], dtype=float)
            n_down = np.array([(arr_down > e).sum() for e in e_axis], dtype=float)
 
            valid = n_up >= 5
            if not valid.any():
                continue
 
            survival = np.full(len(e_axis), np.nan)
            survival[valid] = n_down[valid] / n_up[valid]
 
            err = np.full(len(e_axis), np.nan)
            good = valid & (n_down > 0)
            err[good] = survival[good] * np.sqrt(
                1.0 / n_up[good] + 1.0 / n_down[good])
 
            v = ~np.isnan(survival)
            ax_ratio.plot(e_axis[v], survival[v],
                          color=col, lw=1.8,
                          label=f"{lbl_up} → {lbl_down}")
 
            # Log-space error band: upper/lower = S * exp(±σ_S/S)
            # Symmetric in log space, never clips to zero, and the
            # band width is visually constant regardless of S magnitude —
            # which is exactly what you want on a log y-axis.
            rel_err = np.full(len(e_axis), np.nan)
            rel_err[good] = np.sqrt(1.0 / n_up[good] + 1.0 / n_down[good])
            s_lo = survival * np.exp(-rel_err)
            s_hi = survival * np.exp( rel_err)
            ax_ratio.fill_between(e_axis[v], s_lo[v], s_hi[v],
                                   color=col, alpha=0.20)
 
            in_span, x0 = False, None
            for j, flag in enumerate(~valid):
                if flag and not in_span:
                    x0 = e_axis[j]; in_span = True
                elif not flag and in_span:
                    ax_ratio.axvspan(x0, e_axis[j],
                                     color="#e74c3c", alpha=0.08, zorder=0)
                    in_span = False
            if in_span:
                ax_ratio.axvspan(x0, e_axis[-1],
                                 color="#e74c3c", alpha=0.08, zorder=0)
 
    ax_ratio.axhline(1.0, color="0.3", lw=1.0, ls="--",
                     label="S = 1  (no absorption)")
 
    from matplotlib.patches import Patch
    handles, labels = ax_ratio.get_legend_handles_labels()
    handles.append(Patch(facecolor="#e74c3c", alpha=0.25))
    labels.append("< 5 upstream counts  (unreliable)")
    ax_ratio.legend(handles=handles, labels=labels,
                    fontsize=8.5, framealpha=0.9, ncol=2)
 
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
 
    # Set y-limits from the survival fraction values themselves,
    # not the error bands — the bands can be huge at high E where
    # statistics are poor and would otherwise dominate the axis range.
    all_survival_vals = []
    for line in ax_ratio.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y) & (y > 0)]
        if len(y):
            all_survival_vals.extend(y.tolist())
    if all_survival_vals:
        s_min = max(min(all_survival_vals) * 0.999, 1e-4)
        s_max = min(max(all_survival_vals) * 1.001, 2.0)
        ax_ratio.set_ylim(s_min, s_max)
 
    ax_ratio.set_xlabel("Energy threshold  E  [MeV]")
    ax_ratio.set_ylabel(r"Survival fraction  $S(E)$  [log scale]")
    ax_ratio.set_title(
        "Cumulative survival fraction  (detector stages only — Gun excluded)\n"
        r"$S(E) < 1$: muons absorbed;  log scale = uniform relative errors  |  "
        "red = < 5 upstream counts")
    ax_ratio.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax_ratio.grid(True, which="both", alpha=0.2)
 
    fig.suptitle(
        "Muon energy landscape through detector stack\n"
        "Top: dN/dE spectra per stage    Bottom: cumulative survival fraction S(E)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "energy_loss_cascade", fmt)


def fig_coincidence_overview(coinc_df, coinc_sum, n_events, out, fmt):
    """
    Four-panel coincidence overview corresponding to the real experiments.

    Top-left  — Fold multiplicity: 1/2/3/4-fold event counts (log scale),
                split by whether the Al plate also fired.
                → plateau+coincidence and flux measurement result.

    Top-right — All 15 bitmask patterns ranked by count (horizontal log bar).
                Shows which plate combinations dominate.

    Bottom-left — 6 pair coincidence rates: each bar is N(events where
                  both plates fired, regardless of other plates).
                  → mirrors the PMT-pair coincidence sweep.

    Bottom-right — Plate efficiency per plate X:
                   ε_X = N(4-fold) / N(3-fold trigger excl. X)
                   with binomial ±1σ error bars.
                   → mirrors the real efficiency measurement.
    """
    if coinc_df.empty and not coinc_sum:
        print("  skipping coincidence overview: no data.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_fold, ax_pat), (ax_pair, ax_eff) = axes

    fold_colours = {1: "#aec6e8", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}

    # ── Top-left: fold multiplicity ──────────────────────────────────────────
    if not coinc_df.empty:
        for fold in range(1, 5):
            sub    = coinc_df[coinc_df["n_fired"] == fold]
            sub_al = sub[sub["al_fired"] == 1] if "al_fired" in sub.columns \
                     else pd.DataFrame()
            n, n_al = len(sub), len(sub_al)
            ax_fold.bar(fold, n, color=fold_colours[fold],
                        edgecolor="0.2", linewidth=0.6,
                        label=f"{fold}-fold  N={n:,}", zorder=3)
            if n_al > 0:
                ax_fold.bar(fold, n_al, color=fold_colours[fold],
                            hatch="///", edgecolor="0.2",
                            linewidth=0.6, alpha=0.55, zorder=4,
                            label=f"+Al  N={n_al:,}")
            if n > 0:
                ax_fold.text(fold, n * 1.15, f"{n:,}",
                             ha="center", va="bottom", fontsize=8)
        ax_fold.set_yscale("log")
        ax_fold.set_xticks([1, 2, 3, 4])
        ax_fold.set_xticklabels(["1-fold\n(single)", "2-fold\n(pair)",
                                  "3-fold\n(triple)", "4-fold\n(all)"])
        ax_fold.set_ylabel("Events (log scale)")
        ax_fold.set_title("Coincidence fold multiplicity\n"
                           "(hatched = Al plate also fired above threshold)")
        ax_fold.legend(fontsize=7.5, ncol=2, loc="upper right")
        ax_fold.grid(axis="y", alpha=0.3)

    # ── Top-right: all 15 patterns ───────────────────────────────────────────
    if "patterns" in coinc_sum:
        pat = coinc_sum["patterns"].copy().sort_values("n_events",
                                                        ascending=True)
        n_bits = pat["mask"].astype(int).apply(lambda m: bin(m).count("1"))
        colours = [SCINT_COLOURS[min(b - 1, 3)] for b in n_bits]
        bars = ax_pat.barh(range(len(pat)), pat["n_events"],
                           color=colours, edgecolor="0.3",
                           linewidth=0.4, zorder=3)
        ax_pat.set_yticks(range(len(pat)))
        ax_pat.set_yticklabels(pat["label"], fontsize=8)
        ax_pat.set_xscale("log")
        ax_pat.set_xlabel("Events (log scale)")
        ax_pat.set_title("All coincidence patterns  (colour = fold)")
        ax_pat.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, pat["n_events"]):
            ax_pat.text(bar.get_width() * 1.05,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(val):,}", va="center", fontsize=7)

    # ── Bottom-left: pair coincidence counts ─────────────────────────────────
    if "pairs" in coinc_sum:
        pairs = coinc_sum["pairs"]
        x = np.arange(len(pairs))
        ax_pair.bar(x, pairs["n_events"], color=SCINT_COLOURS[1],
                    edgecolor="0.3", linewidth=0.5, zorder=3)
        # Poisson error
        err = np.sqrt(np.maximum(pairs["n_events"].values, 1))
        ax_pair.errorbar(x, pairs["n_events"], yerr=err,
                         fmt="none", color="0.2", lw=1.0,
                         capsize=3, zorder=4)
        ax_pair.set_xticks(x)
        ax_pair.set_xticklabels(pairs["pair"], rotation=30, ha="right")
        ax_pair.set_ylabel("Events (includes higher folds)")
        ax_pair.set_title("Pair coincidence counts\n"
                           "(any event where both plates of the pair fired)")
        ax_pair.grid(axis="y", alpha=0.3)
        for xi, val in zip(x, pairs["n_events"]):
            ax_pair.text(xi, val * 1.01, f"{int(val):,}",
                         ha="center", va="bottom", fontsize=7.5)

    # ── Bottom-right: plate efficiency ───────────────────────────────────────
    if "efficiency" in coinc_sum:
        eff = coinc_sum["efficiency"]
        x = np.arange(len(eff))
        ax_eff.bar(x, eff["efficiency"], color=_AL_COL,
                   edgecolor="0.3", linewidth=0.5, zorder=3)
        ax_eff.errorbar(x, eff["efficiency"], yerr=eff["efficiency_err"],
                        fmt="none", color="0.2", lw=1.2,
                        capsize=4, zorder=4)
        ax_eff.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax_eff.set_xticks(x)
        ax_eff.set_xticklabels(eff["plate"], rotation=20, ha="right")
        ax_eff.set_ylim(0, 1.15)
        ax_eff.set_ylabel("Efficiency  =  N(4-fold) / N(3-fold trigger)")
        ax_eff.set_title("Plate detection efficiency\n"
                          "(trigger = other 3 plates; ±binomial 1σ)")
        ax_eff.grid(axis="y", alpha=0.3)
        for xi, (_, row) in zip(x, eff.iterrows()):
            ax_eff.text(xi, row["efficiency"] + 0.025,
                        f"{row['efficiency']:.3f}",
                        ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(
        f"Coincidence overview  —  {n_events:,} simulated events\n"
        f"Discriminator threshold: {0.5} MeV per plate",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "coincidence_overview", fmt)


def fig_coincidence_edep(coinc_df, coinc_sum, out, fmt):
    """
    Two-panel energy deposit figure for coincident events.

    Left  — dN/dE distributions for each plate broken down by fold.
             Line style encodes fold (dotted=1, dashed=2, dashdot=3, solid=4).
             Colour encodes plate.  Shows the Landau MIP peak and the
             low-edep tail from stopping muons.

    Right — Mean edep per plate when that plate was part of a coincidence
            (from coincidence_summary.csv).
    """
    if coinc_df.empty:
        print("  skipping coincidence edep: no coincidence data.")
        return

    fig, (ax_dist, ax_mean) = plt.subplots(1, 2, figsize=(13, 5))

    edep_cols   = ["edep_s0_MeV", "edep_s1_MeV",
                   "edep_s2_MeV", "edep_s3_MeV"]
    plate_lbls  = ["Scint 0", "Scint 1", "Scint 2", "Scint 3"]
    fold_styles = {1: (":",  0.7), 2: ("--", 0.9),
                   3: ("-.", 1.1), 4: ("-",  1.5)}

    for ci, (col, lbl) in enumerate(zip(edep_cols, plate_lbls)):
        if col not in coinc_df.columns:
            continue
        for fold, (ls, lw) in fold_styles.items():
            sub = coinc_df[(coinc_df["n_fired"] == fold) &
                           (coinc_df[col] > 0)][col]
            if len(sub) < 5:
                continue
            hi   = min(sub.quantile(0.99) * 1.1, 60.0)
            bins = np.linspace(0, hi, 60)
            counts, edges = np.histogram(sub, bins=bins)
            widths  = np.diff(edges)
            centres = 0.5 * (edges[:-1] + edges[1:])
            mask = counts > 0
            ax_dist.step(centres[mask], (counts / widths)[mask],
                         where="mid", lw=lw, ls=ls,
                         color=SCINT_COLOURS[ci], alpha=0.8,
                         label=f"{lbl} {fold}-fold")

    ax_dist.set_xlabel("Energy deposit  [MeV]")
    ax_dist.set_ylabel("dN/dE  [events / MeV]")
    ax_dist.set_yscale("log")
    ax_dist.set_title("Edep distributions by plate and fold\n"
                       "(colour = plate;  line style = fold)")
    ax_dist.legend(fontsize=6.5, ncol=2, framealpha=0.8)
    ax_dist.grid(True, which="both", alpha=0.2)

    if "edep" in coinc_sum:
        ed = coinc_sum["edep"]
        x  = np.arange(len(ed))
        ax_mean.bar(x, ed["mean_edep_MeV"],
                    color=SCINT_COLOURS[:len(ed)],
                    edgecolor="0.3", linewidth=0.5, zorder=3)
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels(ed["plate"], rotation=20, ha="right")
        ax_mean.set_ylabel("Mean edep  [MeV]")
        ax_mean.set_title("Mean edep per plate\n"
                           "(when that plate participated in a coincidence)")
        ax_mean.grid(axis="y", alpha=0.3)
        for xi, (_, row) in zip(x, ed.iterrows()):
            ax_mean.text(xi, row["mean_edep_MeV"] * 1.02,
                         f"{row['mean_edep_MeV']:.2f} MeV",
                         ha="center", va="bottom", fontsize=9)

    fig.suptitle("Energy deposits in coincident events", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "coincidence_edep", fmt)


def fig_coincidence_rates(coinc_sum, n_events, out, fmt):
    """
    Rate summary bar chart — all named coincidence classes with Poisson
    error bars.  Y-axis is rate per simulated event (dimensionless).
    Multiply by the real muon event rate [Hz] to get the detector rate [Hz].
    Directly analogous to the flux measurement output.
    """
    if "classes" not in coinc_sum:
        print("  skipping coincidence rates: no class data.")
        return

    cls = coinc_sum["classes"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x      = np.arange(len(cls))
    rates  = cls["rate_per_event"].values
    n_evts = cls["n_events"].values
    err    = np.sqrt(np.maximum(n_evts, 1)) / max(n_events, 1)

    colours = [SCINT_COLOURS[min(i, 3)] for i in range(len(cls))]
    ax.bar(x, rates, color=colours, edgecolor="0.3",
           linewidth=0.5, zorder=3)
    ax.errorbar(x, rates, yerr=err, fmt="none",
                color="0.2", lw=1.0, capsize=4, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cls["class"], rotation=20, ha="right")
    ax.set_ylabel("Rate per simulated event")
    ax.set_yscale("log")
    ax.set_title("Coincidence class rates  (flux measurement analogue)\n"
                 "Multiply by muon event rate [Hz] → detector rate [Hz]")
    ax.grid(axis="y", which="both", alpha=0.3)
    for xi, (rate, n) in enumerate(zip(rates, n_evts)):
        ax.text(xi, rate * 1.6, f"N={int(n):,}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    _save(fig, out, "coincidence_rates", fmt)


def main():
    parser = argparse.ArgumentParser(description="Plots for CosmicMuonSim output.")
    parser.add_argument("--dir", "-d", default="results")
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--fmt",    default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--ntraj",  type=int, default=12)
    parser.add_argument("--proj",   default="xz", choices=["xy", "xz"])
    args = parser.parse_args()

    _apply_style()
    d   = Path(args.dir)
    out = d / "plots" if args.save else None

    if not d.exists():
        print(f"Error: '{d}' not found.")
        sys.exit(1)

    edep      = load_edep_summary(d)
    stop      = load_stopping(d)
    stop_cond = load_stopping_conditional(d)
    flu       = load_fluence(d)
    epe       = load_edep_per_event(d)
    steps     = load_steps(d)
    traj      = load_trajectories(d)
    entry_ke  = load_entry_ke(d)
    gun       = load_gun_energy(d)
    decay     = load_decay(d)
    decay_sum = load_decay_summary(d)
    dec_sec   = load_decay_secondary_stopping(d)
    dec_sec_ke = load_decay_secondary_entry_ke(d)
    coinc     = load_coincidences(d)
    coinc_sum = load_coincidence_summary(d)

    # Infer total event count for rate calculations
    n_events = 0
    if not edep.empty:
        nonzero = edep[edep["mean_edep_MeV"] > 0]
        if not nonzero.empty:
            row = nonzero.iloc[0]
            n_events = round(row["total_edep_MeV"] / row["mean_edep_MeV"])

    if edep.empty and not epe.empty:
        print("  Deriving edep_summary from edep_per_event.csv ...")
        edep = derive_edep_summary(epe)

    found = []
    for name, obj in [("edep_summary", edep), ("stopping", stop),
                      ("stopping_conditional", stop_cond),
                      ("fluence", flu), ("edep_per_event", epe),
                      ("steps", steps), ("trajectories", traj),
                      ("entry_ke", entry_ke), ("gun_energy", gun),
                      ("decay", decay), ("decay_summary", decay_sum),
                      ("decay_secondary_stopping", dec_sec),
                      ("decay_secondary_entry_ke", dec_sec_ke),
                      ("coincidences", coinc)]:
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            found.append(name)
        elif isinstance(obj, dict) and obj:
            found.append(name)
    if coinc_sum:
        found.append("coincidence_summary")
    print(f"Loaded: {', '.join(found)}")

    fmt = args.fmt
    fig_detector_schematic(out, fmt)
    fig_edep_by_volume(edep, out, fmt)
    fig_edep_stacked(edep, out, fmt)
    fig_edep_distributions(epe, out, fmt)
    fig_stopping(stop, stop_cond, out, fmt)
    fig_population_flow(stop, out, fmt)
    fig_fluence(flu, out, fmt)
    fig_dedx_profiles(steps, out, fmt)
    fig_hit_positions(steps, out, fmt)
    fig_entry_ke_distributions(entry_ke, out, fmt)
    fig_entry_ke_overlay(entry_ke, out, fmt)
    fig_al_entry_ke(entry_ke, out, fmt)
    fig_ke_spectra(steps, out, fmt)
    fig_process_breakdown(steps, out, fmt)
    fig_gun_energy(gun, out, fmt)
    fig_energy_loss_cascade(entry_ke, gun, out, fmt)
    fig_coincidence_overview(coinc, coinc_sum, n_events, out, fmt)
    fig_coincidence_edep(coinc, coinc_sum, out, fmt)
    fig_coincidence_rates(coinc_sum, n_events, out, fmt)
    fig_decay_overview(decay, decay_sum, out, fmt)
    fig_decay_detail(decay, out, fmt)
    fig_decay_al_plate(decay, out, fmt)
    fig_decay_secondary_stopping(dec_sec, out, fmt)
    fig_decay_secondary_entry_ke(dec_sec_ke, out, fmt)
    fig_trajectories(steps, traj, out, fmt,
                     n_events=args.ntraj, projection=args.proj)

    n_figs = len(plt.get_fignums())
    print(f"\nGenerated {n_figs} figures.")
    if out:
        print(f"Saved to {out}/")
    if not args.no_show and not args.save:
        plt.show()


if __name__ == "__main__":
    main()