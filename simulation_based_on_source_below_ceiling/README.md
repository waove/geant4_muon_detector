# Muon Lifetime Simulation — LUT-Ready Stripped Build

## What changed from the full simulation

| Component               | Full sim                         | This build                        |
|-------------------------|----------------------------------|-----------------------------------|
| Muon source             | Gaisser parametrization at 8 m   | CSV-sampled at coil top (+100 mm) |
| Concrete ceiling/roof   | Present                          | Removed                           |
| Sensitive detectors     | 4 × ScintillatorSD               | Removed (SteppingAction handles)  |
| Output files            | ~10 CSV files + summaries        | 2 event files + 1 summary         |
| B field                 | 40 mT +z in air hole             | Unchanged                         |
| Geometry                | Coil + scints + Al + concrete    | Coil + scints + Al                |
| Polarization            | Not tracked                      | Read from CSV, set on gun         |

## Source CSV format

```
eventID,trackID,pdg,ke_MeV,px,py,pz,polx,poly,polz
0,1,13,245.3,-0.98,0.12,0.15,0.0,0.0,-1.0
```

Place the file as `muon_source.csv` in the run directory (or change the
path in `ActionInitialization.cc`).

## Output files

### `results/scint_hits.csv` — LUT lookup keys + timestamp

One row per particle entering a scintillator plate:

| Column           | Description                                            |
|------------------|--------------------------------------------------------|
| `event_id`       | Event number                                           |
| `plate`          | Scintillator index 0–3 (0 = top at +74 mm)            |
| `pdg`            | PDG code (13, -13, 11, -11, ...)                       |
| `track_id`       | Geant4 track ID                                        |
| `ke_MeV`         | Kinetic energy at plate entry [MeV]                    |
| `is_stopped`     | 1 if particle entered but never exited this plate      |
| `time_ns`        | Global time at entry [ns]                              |
| `is_decay_product`| 1 if parent was a muon that decayed                   |

**LUT mapping:** Use `(|pdg|, ke_MeV, is_stopped)` to look up the PMT
pulse height distribution.  Use `time_ns` for the lifetime histogram
(Δt = t_decay_electron − t_muon for each event).

### `results/coincidence.csv` — edep-based cross-check

One row per event where ≥1 plate exceeded the 0.5 MeV threshold:

| Column      | Description                              |
|-------------|------------------------------------------|
| `event_id`  | Event number                             |
| `coinc_mask`| 4-bit bitmask (bit i = Scint_i fired)   |
| `n_fired`   | Number of plates above threshold (1–4)   |
| `edep*_MeV` | Total energy deposit per plate [MeV]     |

### `results/coincidence_summary.csv` — run-level statistics

Same content as the full sim: fold rates, pair rates, plate
efficiencies (binomial errors), mean edep per plate in coincidences,
all 15 bitmask patterns.

## Decay products born inside scintillators

If a muon stops and decays **inside** a scintillator plate, the decay
e± is created at the decay vertex — it never crosses a geometric
boundary.  `TrackingAction::PreUserTrackingAction` catches this case
and records the "entry" at birth position/time with `is_decay_product=1`.

This is critical for the lifetime measurement: without it, events where
the muon decays inside a scint (rather than in the Al plate) would be
invisible to the stop-signal logic.

## Geometry layout (x-axis is vertical, muons come from +x)

```
x [mm]
+100  ── source plane ──────────────────
 +95  ── copper coil top face ──────────
 +74  │  Scint_0  (20 mm thick)        │
 +54  │  Scint_1  (20 mm thick)        │
  +4  │  Al plate (10 mm thick)        │
 -56  │  Scint_2  (20 mm thick)        │
 -76  │  Scint_3  (20 mm thick)        │
 -95  ── copper coil bottom face ───────
```

B field: 40 mT along +z (transverse to beam → spin precession in x-y).
