# DICT_Physics — Physics-informed MRI sequence classifier

Classifies MRI scans using DICOM acquisition parameters (TR, TE, TI, ETL, FA,
field strength, diffusion tags) rather than series-description keywords.

---

## Output columns

| Column | Values | Notes |
|---|---|---|
| `phys_sequence` | T1W, T2W, T2\*, PD, DWI, Localizer, Unknown, Unknown_GRE | Primary contrast type |
| `phys_acquisition` | FSE, SE, GRE, IR, FLAIR, "" | Pulse sequence family — **info only**, not used in comparisons |
| `phys_fat_sat` | FS, STIR, "" | STIR = IR-based fat suppression; FS = chemical fat suppression |
| `phys_contrast` | Contrast, "" | Gadolinium contrast agent detected |

---

## Classification steps (mutually exclusive, in order)

### Step 0 — Localizer (keyword)
Series description matches `CAL`, `LOC`, `LOCAL`, `SCOUT`, `SURVEY`, or `CALIBRATION`.
→ `Localizer` — exits immediately, no physics evaluation.

---

### Step 1 — DWI
Triggered when **b-value > 0** is present, or `DIFFUSION` appears in `ImageType`.
b = 0 images (non-diffusion-weighted reference volumes) are **not** classified as DWI.

---

### Step 2 — Inversion Recovery (TI > 0)

| Condition | Result |
|---|---|
| 1.5T + TI 110–190 ms | T2W / acquisition = IR / fat_sat = STIR |
| 3T + TI 150–240 ms | T2W / acquisition = IR / fat_sat = STIR |
| 1.5T + TI 1900–2600 ms | T2W / acquisition = FLAIR |
| 3T + TI 2400–3200 ms | T2W / acquisition = FLAIR |
| TE ≤ 30 ms and TR < 4000 ms | T1W / acquisition = IR |
| Otherwise | Unknown / acquisition = IR |

STIR and FLAIR are both mapped to **T2W** with their acquisition type recorded separately.

---

### Step 3 — GRE (`ScanningSequence` contains `GR`)

| Condition | Result |
|---|---|
| TR ≤ 8 ms and TE ≤ 4 ms and FA 35–100° | Localizer |
| TR ≤ 20 ms and TE ≤ 6 ms | T1W / GRE |
| TE ≥ 11 ms | T2\* / GRE |
| Otherwise | Unknown / GRE |

---

### Step 4 — (F)SE (all remaining)

TE is the primary determinant; TR breaks ties.

| TE | TR | Result |
|---|---|---|
| ≤ 25 ms | ≤ 1100 ms | T1W |
| ≤ 25 ms | > 1100 ms | PD |
| ≥ 70 ms | any | T2W |
| 25–70 ms | < 1500 ms | T1W |
| 25–70 ms | ≥ 1500 ms | T2W |

`phys_acquisition` is `FSE` when ETL ≥ 2, otherwise `SE`.

---

## Fat saturation (`phys_fat_sat`)

Detection uses **token matching** on `SeriesDescription` and `ScanOptions`:

| Token / pattern | Detected as |
|---|---|
| FS, FATSAT, FATSUPP, SPIR, SPAIR, FAT, CHEMSAT | FS (chemical) |
| STIR | STIR (IR-based) |
| FAT_SAT, FAT-SAT | FS (regex fallback) |

**Priority**: when TI places the sequence in the STIR range, `phys_fat_sat = STIR`
regardless of any additional chemical FS option detected.

---

## Contrast (`phys_contrast`)

Detected from (in order):
1. `SeriesDescription` tokens: `GD`, `GAD`, or patterns `+C`, `+CTE`, `CTE`
2. `ContrastBolusAgent` field populated with a non-trivial value (not `NONE`/`NO`/`0`)
3. `ContrastBolusAgent` token matches known agents: `DOTAREM`, `MULTIHANCE`, `GADO`, etc.

---

## Usage

```bash
python label_physics.py input.csv [output.csv]
```

`output.csv` defaults to overwriting `input.csv` if not specified.

---

## Files

| File | Purpose |
|---|---|
| `label_physics.py` | Main classifier — reads CSV, writes four `phys_*` columns |
| `visualise_physics.py` | Distribution plots (modality × fat_sat × contrast) |
| `README.md` | This file |
