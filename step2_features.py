"""
=============================================================
Mind vs Clock — Capstone Analysis Pipeline
Step 2: Feature Engineering
=============================================================
Student:    Anzhelika Simonyan
Supervisor: Arman Asryan
University: American University of Armenia, BS Data Science

WHAT THIS DOES:
    Computes 5 behavioral features per trial per participant.
    These features are the input matrix for the HMM in Step 3.

    Feature 1 — rt_norm:          Normalized response time (z-score within participant)
    Feature 2 — rt_trend:         Rolling 5-trial slope of rt_norm (acceleration signal)
    Feature 3 — choice_entropy:   Rolling 10-trial Shannon entropy of choices
    Feature 4 — perf_delta_norm:  Rolling performance deviation from personal baseline
    Feature 5 — timeout_flag:     Binary indicator of cognitive overload (0 or 1)

HOW TO RUN:
    Make sure step1_cleaning.py has been run first.
    Run: python step2_features.py

OUTPUT FILES:
    featured_trials.csv — clean trials with all 5 features added
=============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load clean data from Step 1 ──────────────────────────────────
print("=" * 60)
print("STEP 2 — FEATURE ENGINEERING")
print("=" * 60)

clean_t = pd.read_csv(f"{OUTPUT_DIR}/clean_trials.csv")

# Sort by participant then trial number — critical for rolling features
clean_t = clean_t.sort_values(
    ["participantId", "trialNumber"]
).reset_index(drop=True)

print(f"\nLoaded {len(clean_t)} clean trial rows")
print(f"Participants: {clean_t['participantId'].nunique()}")

# ─────────────────────────────────────────────────────────────────
# FEATURE 1 — Normalized Response Time
# ─────────────────────────────────────────────────────────────────
# Why: removes individual speed differences so we compare each person
# to their own baseline. A naturally fast person and a slow person
# both get z=0 at their personal typical speed.
print("\n▶ Feature 1: Normalized RT (z-score within participant)")

clean_t["rt_mean"] = clean_t.groupby("participantId")["responseTimeMs"].transform("mean")
clean_t["rt_std"]  = clean_t.groupby("participantId")["responseTimeMs"].transform("std")
clean_t["rt_norm"] = (
    (clean_t["responseTimeMs"] - clean_t["rt_mean"]) /
    clean_t["rt_std"].where(clean_t["rt_std"] > 0, 1)
)

print(f"  Mean: {clean_t['rt_norm'].mean():.3f}  Std: {clean_t['rt_norm'].std():.3f}")
print(f"  Range: {clean_t['rt_norm'].min():.2f} to {clean_t['rt_norm'].max():.2f}")

# ─────────────────────────────────────────────────────────────────
# FEATURE 2 — RT Trend (rolling 5-trial slope)
# ─────────────────────────────────────────────────────────────────
# Why: captures whether the person is speeding up (moving toward
# System 1) or slowing down (maintaining System 2).
# Negative slope = accelerating = approaching strategy shift.
print("\n▶ Feature 2: RT Trend (rolling 5-trial slope)")

def rolling_slope(series, window=5):
    """Linear regression slope over a rolling window."""
    slopes = np.full(len(series), 0.0)
    arr    = series.values
    x      = np.arange(window, dtype=float)
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1: i + 1]
        if np.std(y) > 0:
            slope, *_ = stats.linregress(x, y)
            slopes[i] = slope
    return slopes

clean_t["rt_trend"] = clean_t.groupby("participantId")["rt_norm"].transform(
    lambda s: rolling_slope(s, window=5)
)

print(f"  Mean: {clean_t['rt_trend'].mean():.4f}  Std: {clean_t['rt_trend'].std():.4f}")
print(f"  Range: {clean_t['rt_trend'].min():.3f} to {clean_t['rt_trend'].max():.3f}")

# ─────────────────────────────────────────────────────────────────
# FEATURE 3 — Choice Entropy (rolling 10-trial window)
# ─────────────────────────────────────────────────────────────────
# Why: measures how consistent or erratic choices are.
# System 2 = consistent deliberate choices → low entropy (near 0)
# System 1 = impulsive variable choices → high entropy (near 1)
# Shannon entropy of binary choices: H = -p*log2(p) - (1-p)*log2(1-p)
print("\n▶ Feature 3: Choice Entropy (rolling 10-trial window)")

def binary_entropy(p):
    """Shannon entropy of a binary distribution."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def rolling_entropy(series, window=10):
    """Rolling entropy of binary A/B choices."""
    entropies = np.full(len(series), 0.0)
    arr       = (series.values == "A").astype(float)
    for i in range(len(arr)):
        start       = max(0, i - window + 1)
        p_a         = arr[start: i + 1].mean()
        entropies[i] = binary_entropy(p_a)
    return entropies

clean_t["choice_entropy"] = clean_t.groupby("participantId")["choice"].transform(
    lambda s: rolling_entropy(s, window=10)
)

print(f"  Mean: {clean_t['choice_entropy'].mean():.4f}  Std: {clean_t['choice_entropy'].std():.4f}")
print(f"  Range: {clean_t['choice_entropy'].min():.4f} to {clean_t['choice_entropy'].max():.4f}")

# ─────────────────────────────────────────────────────────────────
# FEATURE 4 — Performance Delta (rolling deviation from baseline)
# ─────────────────────────────────────────────────────────────────
# Why: measures when performance drops below a person's typical level.
# A drop signals cognitive degradation — strategy is breaking down.
print("\n▶ Feature 4: Performance Delta (rolling vs personal baseline)")

clean_t["perf_baseline"] = clean_t.groupby("participantId")["pointsEarned"].transform("mean")
clean_t["perf_rolling"]  = clean_t.groupby("participantId")["pointsEarned"].transform(
    lambda s: s.rolling(window=10, min_periods=1).mean()
)
clean_t["perf_delta"] = clean_t["perf_rolling"] - clean_t["perf_baseline"]

def normalize_series(s):
    """Z-score normalization handling zero std."""
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return s - s.mean()
    return (s - s.mean()) / sd

clean_t["perf_delta_norm"] = clean_t.groupby("participantId")["perf_delta"].transform(
    normalize_series
)

print(f"  Mean: {clean_t['perf_delta_norm'].mean():.4f}  Std: {clean_t['perf_delta_norm'].std():.4f}")
print(f"  Range: {clean_t['perf_delta_norm'].min():.3f} to {clean_t['perf_delta_norm'].max():.3f}")

# ─────────────────────────────────────────────────────────────────
# FEATURE 5 — Timeout Flag
# ─────────────────────────────────────────────────────────────────
# Why: a direct binary indicator of cognitive overload.
# A timeout = the participant ran out of cognitive resources
# to respond before the timer expired.
print("\n▶ Feature 5: Timeout flag (binary 0/1)")
clean_t["timeout_flag"] = clean_t["timedOut"].astype(int)
print(f"  Timeout rate: {clean_t['timeout_flag'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────
# QUALITY CHECK
# ─────────────────────────────────────────────────────────────────
feature_cols = ["rt_norm", "rt_trend", "choice_entropy",
                "perf_delta_norm", "timeout_flag"]

print(f"\n{'=' * 60}")
print("FEATURE MATRIX SUMMARY")
print(f"{'=' * 60}")
print(clean_t[feature_cols].describe().round(4).to_string())

print("\nNaN check:")
all_clean = True
for col in feature_cols:
    n = clean_t[col].isna().sum()
    status = "✓" if n == 0 else "⚠ WARNING"
    print(f"  {col}: {n} NaNs  {status}")
    if n > 0:
        all_clean = False
        clean_t[col] = clean_t[col].fillna(0)  # safe fill for HMM

if all_clean:
    print("  All features clean — no NaNs found")

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
# Remove helper columns before saving
drop_cols = ["rt_mean", "rt_std", "perf_baseline", "perf_rolling", "perf_delta"]
clean_t = clean_t.drop(columns=[c for c in drop_cols if c in clean_t.columns])

clean_t.to_csv(f"{OUTPUT_DIR}/featured_trials.csv", index=False)

print(f"\n✓ File saved: {OUTPUT_DIR}/featured_trials.csv")
print(f"  Shape: {clean_t.shape}")
print(f"  New columns added: {feature_cols}")
print("\n→ Run step3_hmm.py next")
