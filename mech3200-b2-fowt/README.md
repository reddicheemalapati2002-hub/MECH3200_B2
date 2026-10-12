# MECH3200 Group Consulting Assignment B2 — Floating Offshore Wind Turbine

Vennkkata Reddi Cheemalapati (46555991, SN* = 91), The University of Queensland, Semester 2 2026.

Scripts, results and geometry supporting the B2 report *Tower Design, Flexible-Base Dynamics and Finite-Element Model of the Floating Offshore Wind Turbine*. Every number quoted in the report is produced by the code in this repository; the commit tagged `submission` is the version used.

## Contents

| File | Purpose |
|---|---|
| `analytical.py` | B2.1–B2.3: cantilever and tip-mass characteristic equations, modal forced response, tower diameter and wall-thickness sizing, nacelle-mass redesign. Writes `analytical.json`. |
| `fe_model.py` | B2.4: planar finite-element model — Timoshenko beam elements for the tower, geometric stiffness for the 400 kN power cable, rigid platform on the four A2 mounts via master–slave constraints, nacelle point mass; modal analysis, forced tip response, validation against the closed-form limits, sensitivity runs. Writes `fe_results.json` and the `fe_*.png` figures. |
| `make_step.py` | Generates the STEP geometry of the designed configuration (`geometry/`). Geometry only — no mesh, loads or boundary conditions. |
| `analytical.json`, `fe_results.json` | Numerical output used in the report tables. |
| `figures/` | All report figures. |
| `geometry/` | `MECH3200_B2_46555991.step` (full assembly), `B2_tower_only.step`. |
| `report/` | LaTeX source of the report and the figures it includes. |

## Reproducing the results

```bash
pip install -r requirements.txt
python analytical.py      # B2.1–B2.3  -> analytical.json, tower_modes_analytical.png
python fe_model.py        # B2.4       -> fe_results.json, fe_modes.png, fe_tower_mode1.png, fe_frf.png
python make_step.py       # optional   -> STEP files (needs cadquery)
cd report && latexmk -pdf MECH3200_B2_46555991.tex
```

Run `analytical.py` before `fe_model.py`; the FE script reads the design values from `analytical.json`.

## Key results (SN* = 91)

- Bare tower for f1 ≥ 1.2·3P at 90.7 m/s tip speed: d_t = 11.83 m; wall from 0.7 % L_t tip amplitude: t = 68.9 mm.
- Nacelle mass lowers f1 by 20.4 % (exact) vs 21.1 % (Rayleigh); redesign d_t = 14.37 m.
- Flexible base: fundamental drops from 0.357 Hz (fixed) to 0.136 Hz (coupled pitch–bending); 3P crosses it at U∞ ≈ 6.9 m/s.
- FE validation: fixed-base, tensioned-cable and platform limits reproduce the closed-form results to four significant figures.
- Nacelle TMD on the coupled mode: m_a = 502 t, k_a = 367 kN/m.
