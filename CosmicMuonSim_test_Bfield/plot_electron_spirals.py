#!/usr/bin/env python3
"""
plot_electron_spirals.py — Visualise electron trajectories for B-field verification.

Reads trajectories.csv from the electron-source diagnostic run and
produces several views of the helical paths to confirm that the
magnetic field is active and correctly oriented.

Expected physics:
  B = 1 T along +z   (UNPHYSICAL — test only)
  Electron Larmor radius: r = p / (0.3 × B[T])  [m, p in GeV/c]
  0.5 MeV e⁻ → r ≈  2.5 mm   (tight spiral)
  3.0 MeV e⁻ → r ≈ 11.6 mm   (wider arc)

  Source in 45 mm air gap between Al plate and Scint 2.
  Electrons shoot along +y and spiral in x-y while drifting.

  Motion in x-y plane: circular (Lorentz force ⊥ B)
  Motion along z: free drift (no force along B)
  → helical trajectories with axis along z

If B-field is OFF: trajectories will be straight lines.
If B-field is ON:  trajectories will show clear curvature/spirals
                   in the x-y projection, straight in z vs time.

Usage:
    python plot_electron_spirals.py --dir results --save
    python plot_electron_spirals.py --dir results              # interactive
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

B_TESLA = 1.0            # 1 T  (unphysical test value)
ME_GEV  = 0.000511       # electron mass in GeV
E_MAX   = 3.0            # max source energy in MeV

# Detector geometry (half-dimensions) for background drawing
_COMPONENTS = [
    (+74,  10,  314, 500,  "Scint 0",  "#d4edda", "#28a745"),
    (+54,  10,  314, 500,  "Scint 1",  "#d4edda", "#28a745"),
    (-56,  10,  314, 500,  "Scint 2",  "#d4edda", "#28a745"),
    (-76,  10,  314, 500,  "Scint 3",  "#d4edda", "#28a745"),
    (+4,   5,   314, 500,  "Al plate", "#cce5ff", "#0069d9"),
    (0,    95,  315, 360,  "Cu encl.", "#fdf2f2", "#c0392b"),
]

KE_CMAP = plt.cm.plasma


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 9, "legend.framealpha": 0.9,
        "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })


def _draw_detector_bg(ax, projection="xy", show_labels=True):
    """Draw detector rectangles on a 2D axis."""
    for xc, hx, hy, hz, lbl, fc, ec in _COMPONENTS:
        if projection == "xy":
            w, h = 2 * hx, 2 * hy
            corner = (xc - hx, -hy)
        elif projection == "xz":
            w, h = 2 * hx, 2 * hz
            corner = (xc - hx, -hz)
        elif projection == "yz":
            w, h = 2 * hy, 2 * hz
            corner = (-hy, -hz)
        else:
            continue

        is_cu = lbl.startswith("Cu")
        ax.add_patch(plt.Rectangle(
            corner, w, h,
            facecolor=fc, edgecolor=ec,
            linewidth=1.2 if is_cu else 0.8,
            linestyle="--" if is_cu else "-",
            zorder=0 if is_cu else 1, alpha=0.35))
        if show_labels and not is_cu:
            if projection == "xy":
                ax.text(xc, hy - 15, lbl, ha="center", va="top",
                        fontsize=6.5, color=ec, zorder=5)
            elif projection == "xz":
                ax.text(xc, hz - 15, lbl, ha="center", va="top",
                        fontsize=6.5, color=ec, zorder=5)


def larmor_radius_mm(ke_MeV):
    """Compute Larmor radius in mm for an electron with given KE in MeV."""
    E_total = ke_MeV / 1000.0 + ME_GEV   # total energy in GeV
    p_GeV = np.sqrt(E_total**2 - ME_GEV**2)
    r_m = p_GeV / (0.3 * B_TESLA)
    return r_m * 1000.0   # convert to mm


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectories(d):
    p = d / "trajectories.csv"
    if not p.exists():
        print(f"Error: {p} not found.")
        return pd.DataFrame()
    df = pd.read_csv(p, on_bad_lines="skip")
    rn = {"edep_MeV": "edep", "stepLen_mm": "step_len", "KE_MeV": "KE",
          "x_mm": "x", "y_mm": "y", "z_mm": "z", "time_ns": "time"}
    df = df.rename(columns=rn)
    for col in ["trackID", "parentID", "pdg", "edep", "KE", "x", "y", "z", "time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trackID", "x", "y", "z"])
    print(f"  Loaded {len(df):,} trajectory steps from {df['eventID'].nunique()} events")
    return df


def load_gun_energy(d):
    p = d / "gun_energy.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, on_bad_lines="skip")
    for col in ["pdg", "energy_MeV", "cosTheta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["energy_MeV"])
    print(f"  Loaded {len(df):,} gun energy records")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def fig_spiral_xy(traj, out, fmt, n_events=12):
    """
    x-y projection of electron trajectories.
    This is the KEY diagnostic: spirals in x-y confirm B along z.
    If B is off, tracks will be straight lines in this view.
    """
    if traj.empty:
        return

    electrons = traj[traj["pdg"] == 11].copy()
    if electrons.empty:
        electrons = traj[traj["trackID"] == 1].copy()  # primary tracks

    events = sorted(electrons["eventID"].unique())
    n_show = min(n_events, len(events))
    selected = events[:n_show]

    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.ravel()

    for idx, evtID in enumerate(selected):
        ax = axes_flat[idx]
        _draw_detector_bg(ax, projection="xy", show_labels=(idx == 0))

        evt = electrons[electrons["eventID"] == evtID]
        for trkID, grp in evt.groupby("trackID"):
            grp = grp.sort_values("time")
            xs = grp["x"].values
            ys = grp["y"].values
            ke = grp["KE"].values

            # Colour by kinetic energy
            if len(xs) > 1:
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                ke_norm = ke[:-1] / max(ke.max(), 0.01)
                lc = LineCollection(segments, cmap=KE_CMAP,
                                     norm=plt.Normalize(0, max(ke.max(), 0.1)))
                lc.set_array(ke[:-1])
                lc.set_linewidth(1.5 if trkID == 1 else 0.8)
                ax.add_collection(lc)

            # Mark start position
            ax.plot(xs[0], ys[0], "o", ms=4, color="lime",
                    markeredgecolor="black", markeredgewidth=0.5, zorder=20)

        # Annotate initial KE
        primary = evt[evt["trackID"] == 1]
        if not primary.empty:
            ke0 = primary["KE"].iloc[0]
            r_L = larmor_radius_mm(ke0)
            ax.set_title(f"Event {int(evtID)}\n"
                         f"KE₀ = {ke0:.1f} MeV,  r_L = {r_L:.0f} mm",
                         fontsize=9)

        ax.set_xlim(-50, 12)
        ax.set_ylim(-320, 320)
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("y  [mm]  (drift direction)")
        ax.set_aspect("auto")
        ax.grid(alpha=0.2)

    for idx in range(n_show, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=KE_CMAP,
                                norm=plt.Normalize(0, E_MAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[:n_show].tolist(),
                         shrink=0.6, pad=0.02)
    cbar.set_label("Kinetic energy  [MeV]")

    fig.suptitle(
        "Electron trajectories — x–y projection  (B along z)\n"
        "Spirals confirm active B-field;  straight lines → B is OFF\n"
        "Green dot = source position",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out, "electron_spiral_xy", fmt)


def fig_spiral_xz(traj, out, fmt, n_events=12):
    """
    x-z projection showing helical pitch along B (z-axis).
    Electrons drift along z while spiralling in x-y.
    """
    if traj.empty:
        return

    electrons = traj[traj["pdg"] == 11].copy()
    if electrons.empty:
        electrons = traj[traj["trackID"] == 1].copy()

    events = sorted(electrons["eventID"].unique())
    n_show = min(n_events, len(events))
    selected = events[:n_show]

    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.ravel()

    for idx, evtID in enumerate(selected):
        ax = axes_flat[idx]
        _draw_detector_bg(ax, projection="xz", show_labels=(idx == 0))

        evt = electrons[electrons["eventID"] == evtID]
        for trkID, grp in evt.groupby("trackID"):
            grp = grp.sort_values("time")
            xs = grp["x"].values
            zs = grp["z"].values
            ke = grp["KE"].values

            if len(xs) > 1:
                points = np.array([xs, zs]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=KE_CMAP,
                                     norm=plt.Normalize(0, max(ke.max(), 0.1)))
                lc.set_array(ke[:-1])
                lc.set_linewidth(1.5 if trkID == 1 else 0.8)
                ax.add_collection(lc)

            ax.plot(xs[0], zs[0], "o", ms=4, color="lime",
                    markeredgecolor="black", markeredgewidth=0.5, zorder=20)

        ax.set_xlim(-50, 12)
        ax.set_ylim(-400, 400)
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("z  [mm]   (B-field direction)")
        ax.set_aspect("auto")
        ax.set_title(f"Event {int(evtID)}", fontsize=9)
        ax.grid(alpha=0.2)

    for idx in range(n_show, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Electron trajectories — x–z projection\n"
        "z = B-field direction:  drift along z,  spiral in x-y",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out, "electron_spiral_xz", fmt)


def fig_spiral_3d(traj, out, fmt, n_events=6):
    """
    3D view of electron helices.
    """
    if traj.empty:
        return

    electrons = traj[traj["pdg"] == 11].copy()
    if electrons.empty:
        electrons = traj[traj["trackID"] == 1].copy()

    events = sorted(electrons["eventID"].unique())
    n_show = min(n_events, len(events))
    selected = events[:n_show]

    ncols = min(3, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

    for idx, evtID in enumerate(selected):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        evt = electrons[electrons["eventID"] == evtID]
        for trkID, grp in evt.groupby("trackID"):
            grp = grp.sort_values("time")
            xs = grp["x"].values
            ys = grp["y"].values
            zs = grp["z"].values
            ke = grp["KE"].values

            # Colour segments by KE
            for i in range(len(xs) - 1):
                colour = KE_CMAP(min(ke[i] / 5.0, 1.0))
                ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                        color=colour,
                        linewidth=1.5 if trkID == 1 else 0.6,
                        alpha=0.8)

            ax.scatter(xs[0], ys[0], zs[0], c="lime", s=30,
                       edgecolors="black", linewidths=0.5, zorder=20)

        # Draw B-field direction arrow
        ax.quiver(0, 0, -350, 0, 0, 100,
                  color="red", arrow_length_ratio=0.15,
                  linewidth=2, alpha=0.7)
        ax.text(0, 0, -340, "B", color="red", fontsize=12,
                fontweight="bold", ha="center")

        primary = evt[evt["trackID"] == 1]
        ke0 = primary["KE"].iloc[0] if not primary.empty else 0
        ax.set_title(f"Event {int(evtID)}  (KE₀ = {ke0:.1f} MeV)", fontsize=9)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]  (drift)")
        ax.set_zlabel("z [mm] (B dir)")
        ax.set_xlim(-50, 12)
        ax.set_ylim(-320, 320)
        ax.set_zlim(-400, 400)

    fig.suptitle(
        "3D electron trajectories — helices around B-field axis (z)\n"
        "Red arrow = B direction;  green dot = source",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "electron_spiral_3d", fmt)


def fig_spiral_overlay(traj, out, fmt):
    """
    All primary electron trajectories overlaid on one x-y plot,
    coloured by initial KE.  Shows the range of Larmor radii.
    """
    if traj.empty:
        return

    primaries = traj[(traj["pdg"] == 11) & (traj["trackID"] == 1)].copy()
    if primaries.empty:
        primaries = traj[traj["trackID"] == 1].copy()

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(14, 6))

    events = sorted(primaries["eventID"].unique())
    norm = plt.Normalize(0.5, E_MAX)

    for evtID in events:
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        ke0 = grp["KE"].iloc[0]
        colour = KE_CMAP(norm(ke0))

        ax_xy.plot(grp["x"], grp["y"], color=colour, lw=0.8, alpha=0.7)
        ax_yz.plot(grp["y"], grp["z"], color=colour, lw=0.8, alpha=0.7)

    _draw_detector_bg(ax_xy, projection="xy", show_labels=True)

    ax_xy.set_xlim(-50, 12)
    ax_xy.set_ylim(-320, 320)
    ax_xy.set_xlabel("x  [mm]")
    ax_xy.set_ylabel("y  [mm]  (drift direction)")
    ax_xy.set_title("x–y projection  (perpendicular to B)\n"
                     "Sinusoidal x-oscillation while drifting in y → B is ON")
    ax_xy.set_aspect("auto")
    ax_xy.grid(alpha=0.2)

    ax_yz.set_xlabel("y  [mm]  (drift direction)")
    ax_yz.set_ylabel("z  [mm]  (B direction)")
    ax_yz.set_title("y–z projection\n"
                     "Drift along y, helical pitch in z")
    ax_yz.set_aspect("equal")
    ax_yz.grid(alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=KE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_xy, ax_yz], shrink=0.7, pad=0.02)
    cbar.set_label("Initial kinetic energy  [MeV]")

    fig.suptitle(
        f"All primary electron trajectories ({len(events)} events)\n"
        "Colour = initial KE;  tighter spirals = lower energy = smaller Larmor radius",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out, "electron_spiral_overlay", fmt)


def fig_larmor_analysis(traj, out, fmt):
    """
    Quantitative B-field verification.

    For each primary electron event:
    1. Measure the curvature radius from the x-y trajectory
    2. Compare to the expected Larmor radius from the initial KE
    3. Plot measured vs expected — should lie on y=x if B is correct.

    Method: fit a circle to the first N points of the x-y trajectory.
    """
    if traj.empty:
        return

    primaries = traj[(traj["pdg"] == 11) & (traj["trackID"] == 1)].copy()
    if primaries.empty:
        primaries = traj[traj["trackID"] == 1].copy()

    events = sorted(primaries["eventID"].unique())

    ke_list = []
    r_expected = []
    r_measured = []

    for evtID in events:
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        if len(grp) < 10:
            continue

        ke0 = grp["KE"].iloc[0]
        r_exp = larmor_radius_mm(ke0)

        # Use first 50 points (or all if fewer) to fit a circle in x-y
        xs = grp["x"].values[:80]
        ys = grp["y"].values[:80]

        # Algebraic circle fit: (x-a)² + (y-b)² = r²
        # Linearise: x² + y² = 2ax + 2by + (r² - a² - b²)
        # Let c = r² - a² - b², then: 2ax + 2by + c = x² + y²
        A = np.column_stack([2 * xs, 2 * ys, np.ones(len(xs))])
        b_vec = xs**2 + ys**2

        try:
            result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            a_fit, b_fit, c_fit = result
            r_fit = np.sqrt(c_fit + a_fit**2 + b_fit**2)

            # Sanity check: radius should be positive and not absurdly large
            if 0.5 < r_fit < 500:
                ke_list.append(ke0)
                r_expected.append(r_exp)
                r_measured.append(r_fit)
        except Exception:
            continue

    if len(ke_list) < 3:
        print("  Insufficient data for Larmor radius analysis.")
        return

    ke_arr = np.array(ke_list)
    r_exp_arr = np.array(r_expected)
    r_meas_arr = np.array(r_measured)

    fig, (ax_comp, ax_ratio, ax_ke) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Left: measured vs expected ───────────────────────────────────────
    ax_comp.scatter(r_exp_arr, r_meas_arr, c=ke_arr, cmap=KE_CMAP,
                    s=25, edgecolors="0.3", linewidths=0.4, zorder=3)
    rng = [0, max(r_exp_arr.max(), r_meas_arr.max()) * 1.1]
    ax_comp.plot(rng, rng, "k--", lw=1, label="y = x  (perfect B-field)")
    ax_comp.set_xlabel("Expected Larmor radius  [mm]")
    ax_comp.set_ylabel("Measured Larmor radius  [mm]")
    ax_comp.set_title("Measured vs expected Larmor radius\n"
                       "(points on diagonal → B-field is correct)")
    ax_comp.legend(fontsize=9)
    ax_comp.set_aspect("equal")
    ax_comp.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=KE_CMAP,
                                norm=plt.Normalize(ke_arr.min(), ke_arr.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_comp, shrink=0.7, label="KE [MeV]")

    # ── Centre: ratio measured/expected vs KE ────────────────────────────
    ratio = r_meas_arr / r_exp_arr
    ax_ratio.scatter(ke_arr, ratio, c=ke_arr, cmap=KE_CMAP,
                     s=25, edgecolors="0.3", linewidths=0.4, zorder=3)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=1, label="ratio = 1")
    ax_ratio.set_xlabel("Initial KE  [MeV]")
    ax_ratio.set_ylabel("r_measured / r_expected")
    ax_ratio.set_title("Larmor radius ratio\n"
                        f"(~1.0 means B matches {B_TESLA:.0f} T specification)")
    ax_ratio.set_ylim(0, 2.5)
    ax_ratio.legend()
    ax_ratio.grid(alpha=0.3)

    # Stats box
    med_ratio = np.median(ratio)
    std_ratio = np.std(ratio)
    ax_ratio.text(0.95, 0.95,
                  f"Median ratio: {med_ratio:.3f}\n"
                  f"Std dev: {std_ratio:.3f}\n"
                  f"N events: {len(ratio)}",
                  transform=ax_ratio.transAxes,
                  ha="right", va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            alpha=0.8, ec="0.7"))

    # ── Right: expected Larmor radius vs KE (reference curve) ────────────
    ke_ref = np.linspace(0.5, E_MAX, 200)
    r_ref = np.array([larmor_radius_mm(k) for k in ke_ref])
    ax_ke.plot(ke_ref, r_ref, "k-", lw=2, label="Theory: r = p/(0.3·B)")
    ax_ke.scatter(ke_arr, r_meas_arr, c="tab:blue", s=20, alpha=0.7,
                  edgecolors="0.3", linewidths=0.4, zorder=3,
                  label="Measured (circle fit)")
    ax_ke.set_xlabel("Initial KE  [MeV]")
    ax_ke.set_ylabel("Larmor radius  [mm]")
    ax_ke.set_title("Larmor radius vs kinetic energy\n"
                     "(data should follow theory curve)")
    ax_ke.legend(fontsize=9)
    ax_ke.grid(alpha=0.3)

    fig.suptitle(
        f"Quantitative B-field verification  —  B = {B_TESLA:.0f} T along +z\n"
        f"Circle fit to x-y trajectory of {len(ke_list)} primary electrons",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out, "larmor_analysis", fmt)


def fig_energy_loss(traj, out, fmt):
    """
    KE vs time and vs path length for primary electrons.
    Shows energy loss as electrons spiral through air and scatter
    in detector materials.
    """
    if traj.empty:
        return

    primaries = traj[(traj["pdg"] == 11) & (traj["trackID"] == 1)].copy()
    if primaries.empty:
        primaries = traj[traj["trackID"] == 1].copy()

    events = sorted(primaries["eventID"].unique())
    n_show = min(20, len(events))

    fig, (ax_t, ax_vol) = plt.subplots(1, 2, figsize=(13, 5))
    norm = plt.Normalize(0.5, E_MAX)

    for evtID in events[:n_show]:
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        ke0 = grp["KE"].iloc[0]
        colour = KE_CMAP(norm(ke0))

        ax_t.plot(grp["time"], grp["KE"], color=colour, lw=0.8, alpha=0.7)

        # Cumulative path length
        dx = np.diff(grp["x"].values)
        dy = np.diff(grp["y"].values)
        dz = np.diff(grp["z"].values)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        path = np.concatenate([[0], np.cumsum(ds)])
        ax_vol.plot(path, grp["KE"].values, color=colour, lw=0.8, alpha=0.7)

    ax_t.set_xlabel("Time  [ns]")
    ax_t.set_ylabel("Kinetic energy  [MeV]")
    ax_t.set_title("KE vs time\n(energy loss from ionisation in air + materials)")
    ax_t.grid(alpha=0.3)

    ax_vol.set_xlabel("Path length  [mm]")
    ax_vol.set_ylabel("Kinetic energy  [MeV]")
    ax_vol.set_title("KE vs path length\n(steeper drops = denser material)")
    ax_vol.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=KE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_t, ax_vol], shrink=0.7)
    cbar.set_label("Initial KE  [MeV]")

    fig.suptitle(f"Electron energy loss ({n_show} events)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "electron_energy_loss", fmt)


def fig_bfield_diagnostic(traj, out, fmt):
    """
    Single diagnostic summary figure with clear PASS/FAIL indication.

    Checks:
    1. Are any steps recorded at all?
    2. Do trajectories curve in x-y? (std of x or y spread > threshold)
    3. Is the mean curvature consistent with expected Larmor radius?

    Produces a simple visual verdict.
    """
    if traj.empty:
        return

    primaries = traj[(traj["pdg"] == 11) & (traj["trackID"] == 1)].copy()
    if primaries.empty:
        primaries = traj[traj["trackID"] == 1].copy()

    events = sorted(primaries["eventID"].unique())

    fig, (ax_diag, ax_demo) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: diagnostic verdict ─────────────────────────────────────────
    checks = []

    # Check 1: steps recorded
    n_steps = len(primaries)
    checks.append(("Steps recorded", n_steps > 0,
                    f"N = {n_steps:,}"))

    # Check 2: trajectory curvature
    curvatures = []
    for evtID in events[:30]:
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        if len(grp) < 5:
            continue
        xs = grp["x"].values
        ys = grp["y"].values
        # Measure displacement vs path length
        dx = np.diff(xs)
        dy = np.diff(ys)
        dz = np.diff(grp["z"].values)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        total_path = ds.sum()
        net_disp = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        if total_path > 0:
            ratio = net_disp / total_path   # < 1 means curved
            curvatures.append(ratio)

    if curvatures:
        mean_ratio = np.mean(curvatures)
        is_curved = mean_ratio < 0.9
        checks.append(("Trajectories curve",
                        is_curved,
                        f"displacement/path = {mean_ratio:.3f}\n"
                        f"(< 0.9 = curved, ~1.0 = straight)"))
    else:
        checks.append(("Trajectories curve", False, "No data"))

    # Check 3: Larmor radius order-of-magnitude correct
    r_measured_list = []
    for evtID in events[:30]:
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        if len(grp) < 10:
            continue
        xs = grp["x"].values[:80]
        ys = grp["y"].values[:80]
        A = np.column_stack([2 * xs, 2 * ys, np.ones(len(xs))])
        b_vec = xs**2 + ys**2
        try:
            result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            r_fit = np.sqrt(result[2] + result[0]**2 + result[1]**2)
            if 5 < r_fit < 2000:
                r_measured_list.append(r_fit)
        except Exception:
            pass

    if r_measured_list:
        med_r = np.median(r_measured_list)
        # Expected range: 2.5–12 mm for 0.5–3 MeV at 1 T
        r_ok = 1 < med_r < 30
        checks.append(("Larmor radius plausible",
                        r_ok,
                        f"Median measured: {med_r:.1f} mm\n"
                        f"Expected range: 2.5–12 mm"))
    else:
        checks.append(("Larmor radius plausible", False, "No fits"))

    # Draw verdict
    all_pass = all(ok for _, ok, _ in checks)
    verdict = "✓  B-FIELD ACTIVE" if all_pass else "✗  CHECK B-FIELD"
    verdict_col = "#27ae60" if all_pass else "#c0392b"

    ax_diag.text(0.5, 0.92, "Magnetic Field Diagnostic",
                 transform=ax_diag.transAxes,
                 ha="center", va="top", fontsize=16, fontweight="bold")

    ax_diag.text(0.5, 0.82, verdict,
                 transform=ax_diag.transAxes,
                 ha="center", va="top", fontsize=22, fontweight="bold",
                 color=verdict_col,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white",
                           ec=verdict_col, lw=3))

    for i, (name, ok, detail) in enumerate(checks):
        y = 0.60 - i * 0.18
        symbol = "✓" if ok else "✗"
        col = "#27ae60" if ok else "#c0392b"
        ax_diag.text(0.1, y, f"{symbol}  {name}",
                     transform=ax_diag.transAxes,
                     fontsize=13, color=col, fontweight="bold", va="top")
        ax_diag.text(0.15, y - 0.06, detail,
                     transform=ax_diag.transAxes,
                     fontsize=9, color="0.3", va="top")

    ax_diag.axis("off")

    # ── Right: example trajectory showing spiral ─────────────────────────
    if events:
        evtID = events[0]
        grp = primaries[primaries["eventID"] == evtID].sort_values("time")
        _draw_detector_bg(ax_demo, projection="xy", show_labels=True)
        ax_demo.plot(grp["x"], grp["y"], color="#2980b9", lw=1.5, alpha=0.8)
        ax_demo.plot(grp["x"].iloc[0], grp["y"].iloc[0], "o",
                     ms=6, color="lime", markeredgecolor="black",
                     markeredgewidth=0.8, zorder=20)
        ke0 = grp["KE"].iloc[0]
        ax_demo.set_title(f"Example trajectory  (Event {int(evtID)}, "
                          f"KE₀ = {ke0:.1f} MeV)")
    ax_demo.set_xlim(-50, 12)
    ax_demo.set_ylim(-320, 320)
    ax_demo.set_xlabel("x  [mm]")
    ax_demo.set_ylabel("y  [mm]  (drift direction)")
    ax_demo.set_aspect("auto")
    ax_demo.grid(alpha=0.2)

    fig.suptitle(f"B = {B_TESLA:.1f} T along +z  |  "
                 f"Electron source: 0.5–{E_MAX:.0f} MeV",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out, "bfield_diagnostic", fmt)


def fig_gun_validation(gun, out, fmt):
    """Simple gun energy spectrum to verify the electron source."""
    if gun.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.linspace(0, E_MAX + 0.5, 50)
    ax.hist(gun["energy_MeV"], bins=bins, color="#2980b9",
            edgecolor="0.3", linewidth=0.4, alpha=0.8)
    ax.axvline(0.5, color="red", ls="--", lw=1, label=f"Source range: 0.5–{E_MAX:.0f} MeV")
    ax.axvline(E_MAX, color="red", ls="--", lw=1)
    ax.set_xlabel("Electron kinetic energy  [MeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Gun energy spectrum  (N = {len(gun):,})\n"
                 f"Should be uniform between 0.5 and {E_MAX:.0f} MeV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "electron_gun_energy", fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Save helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir, name, fmt):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise electron spirals for B-field verification.")
    parser.add_argument("--dir", "-d", default="results",
                        help="Results directory (default: results)")
    parser.add_argument("--save", action="store_true",
                        help="Save plots to results/plots/")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--ntraj", type=int, default=12,
                        help="Number of events per trajectory panel")
    args = parser.parse_args()

    _apply_style()
    d = Path(args.dir)
    out = d / "plots" if args.save else None

    if not d.exists():
        print(f"Error: '{d}' not found.")
        sys.exit(1)

    traj = load_trajectories(d)
    gun  = load_gun_energy(d)

    if traj.empty:
        print("\nNo trajectory data found. Run the simulation first:")
        print("  ./CosmicMuonSim -n 200")
        print("Then re-run this script.")
        sys.exit(1)

    n_electrons = len(traj[traj["pdg"] == 11])
    n_events = traj["eventID"].nunique()
    print(f"\n  Total steps: {len(traj):,}")
    print(f"  Events: {n_events}")
    print(f"  Electron steps: {n_electrons:,}")
    print()

    fig_bfield_diagnostic(traj, out, args.fmt)
    fig_spiral_xy(traj, out, args.fmt, n_events=args.ntraj)
    fig_spiral_xz(traj, out, args.fmt, n_events=args.ntraj)
    fig_spiral_3d(traj, out, args.fmt, n_events=min(6, args.ntraj))
    fig_spiral_overlay(traj, out, args.fmt)
    fig_larmor_analysis(traj, out, args.fmt)
    fig_energy_loss(traj, out, args.fmt)
    fig_gun_validation(gun, out, args.fmt)

    n_figs = len(plt.get_fignums())
    print(f"\nGenerated {n_figs} figures.")
    if out:
        print(f"Saved to {out}/")
    if not args.no_show and not args.save:
        plt.show()


if __name__ == "__main__":
    main()
