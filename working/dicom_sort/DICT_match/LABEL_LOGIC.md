# DICOM Label Logic (`label_csv.py`)

Three columns are appended to the DICOM header CSV:
`sequence_type`, `fat_sat`, `contrast`

---

## Pre-filtering

| Step | Rule |
|---|---|
| Drop non-MR | Keep only rows where `modality == "MR"` |
| Drop incomplete | Drop rows missing `repetition_time_ms`, `echo_time_ms`, or `magnetic_field_str` |

---

## Column 1 — `sequence_type`

### Step 1: Keyword matching on `series_description`

Checked in order — **first match wins**.

| Priority | Keywords / Pattern | Label |
|---|---|---|
| 1 | `CAL`, `LOC`, `LOCAL`, `SCOUT`, `SURVEY`, `CALIBRATION` | `localizer` |
| 2 | `DWI`, `DIFF`, `DIFUSION`, `ADC`, `DIFU`, `DIF` | `DWI` |
| 3 | `PERFUSION` | `perfusion` |
| 4 | `LAVA`, `FAME` (whole word) | `T1W` |
| 5 | `STIR` (token) | `T2W` |
| 6 | `T1` | `T1W` |
| 7 | `T2*`, `MERGE` | `T2*` |
| 8 | `T2` or `STIR` | `T2*` if GRE, else `T2W` (see GRE/SE check below) |
| 9 | `PDW` (token) | `PD` |
| 10 | `DP`, `PD` (tokens) | `PD` |
| — | no match | `""` (empty) |

### GRE vs SE disambiguation (used at priority 8)

Tokens are checked in both `series_description` and `scan_options`.
SE markers take priority if both are present.

| Sequence type | Tokens | Result |
|---|---|---|
| Spin Echo (SE) | `SE`, `FSE`, `TSE`, `HASTE`, `RARE`, `CPMG` | → `T2W` |
| Gradient Echo (GRE) | `GRE`, `GE`, `SPGR`, `FLASH`, `FISP`, `FIESTA`, `TRUFI`, `FFE`, `VIBE`, `LAVA` | → `T2*` |

### Step 2: Fallback classification for unmatched rows (`sequence_type == ""`)

Applied before PD reclassification. Determines GRE vs SE via `_is_gre()`, then applies thresholds.

**GRE sequences** (T2* decay is fast, TE thresholds are tight):

| TE | TR | Label |
|---|---|---|
| < 10 ms | < 600 ms | `T1W` |
| < 10 ms | ≥ 600 ms | `PD` |
| > 15 ms | > 1000 ms | `T2*` |
| > 15 ms | ≤ 1000 ms | `mixed` |
| 10 – 15 ms | any | `PD` |

**SE sequences** (T2 decay is slower, TE thresholds are wider):

| TE | TR | Label |
|---|---|---|
| < 25 ms | < 600 ms | `T1W` |
| < 25 ms | > 1500 ms | `PD` |
| > 60 ms | > 1500 ms | `T2W` |
| other | any | `""` (truly ambiguous) |

### Step 3: PD reclassification using TR / TE

Applied after step 1 for all rows initially labelled `PD`.

| TE | TR | Final label |
|---|---|---|
| < 10 ms | < 600 ms | `T1W` |
| < 10 ms | ≥ 600 ms | `PD` |
| > 15 ms | > 1000 ms | `T2*` |
| > 15 ms | ≤ 1000 ms | `mixed` |
| 10 – 15 ms | any | `PD` |
| TR or TE unparseable | — | `PD` (unchanged) |

**Rationale:**
- Short TE suppresses T2* decay → contrast is determined by TR (T1 vs PD)
- Long TE allows T2* decay → long TR removes T1 effects, leaving T2* contrast
- Mixed (short TR + long TE) has both T1 and T2* contamination

---

## Column 2 — `fat_sat`

Checks tokens in both `series_description` and `scan_options`.

| Match | Label |
|---|---|
| Token is one of: `STIR`, `FS`, `FATSAT`, `FATSUPP`, `SPIR`, `SPAIR`, `FLAIR`, `FAT` | `fatsat` |
| No match | `""` |

---

## Column 3 — `contrast`

Checked in order — **first match wins**.

| Priority | Condition | Label |
|---|---|---|
| 1 | `series_description` tokens contain `GD` or `GAD` | `contrast` |
| 1 | `series_description` matches `+C`, `+CTE`, or `CTE` (whole word) | `contrast` |
| 2 | `Contrast_Agent` non-empty **and** `Total_Dose` non-empty / non-zero | `contrast` |
| 3 | `Contrast_Agent` tokens contain `YES`, `Y`, `GD`, `CONTRASTE`, `DOTAREM`, `GADO`, `GAD`, `MH`, `MULTIHANCE` **and** `Total_Dose != "0"` | `contrast` |
| — | none of the above | `""` |
