"""
=============================================================
Mind vs Clock — Capstone Analysis Pipeline
Step 1: Data Cleaning
=============================================================
Student:    Anzhelika Simonyan
Supervisor: Arman Asryan
University: American University of Armenia, BS Data Science

HOW TO RUN:
    1. Place your Excel file in the same folder as this script
    2. Update DATA_FILE below to match your filename
    3. Run: python step1_cleaning.py

OUTPUT FILES:
    clean_participants.csv   — verified participant data
    clean_trials.csv         — verified main trial data
    exclusion_log.csv        — record of every removed participant + reason
=============================================================
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG — update this to match your file ──────────────────────
DATA_FILE   = "Mind_vs_Clock_Full_Dataset.xlsx"
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — DATA CLEANING")
print("=" * 60)

xl = pd.ExcelFile(DATA_FILE)
p  = pd.read_excel(xl, sheet_name="Participants")
t  = pd.read_excel(xl, sheet_name="Trials")

print(f"\nLoaded {len(p)} participants and {len(t)} trial rows")
print(f"Columns (Participants): {list(p.columns)}")
print(f"Columns (Trials):       {list(t.columns)}")

exclusion_log = []

# ── Rule 1: Remove practice trials (block = 0) ───────────────────
# Practice trials are unscored warmup — should not enter the HMM
main_t = t[t["block"] > 0].copy()
print(f"\n✓ Rule 1 — Remove practice trials (block=0)")
print(f"  Removed {len(t) - len(main_t)} practice rows")
print(f"  {len(main_t)} main trial rows remain")

# ── Rule 2: Flag impossibly fast responses (< 200ms) ─────────────
# Under 200ms is physiologically impossible for a deliberate keypress
# These are accidental button presses
fast_mask  = main_t["responseTimeMs"] < 200
fast_count = fast_mask.sum()
main_t["is_too_fast"] = fast_mask
print(f"\n✓ Rule 2 — Flag responses under 200ms")
print(f"  Found {fast_count} impossibly fast responses to remove")

# ── Rule 3: Flag participants with > 20% timeout rate ────────────
# If someone timed out on more than 20% of trials they were
# probably not paying attention — data is unreliable
timeout_rates = main_t.groupby("participantId")["timedOut"].mean()
high_timeout  = timeout_rates[timeout_rates > 0.20]
print(f"\n✓ Rule 3 — Flag participants with >20% timeout rate")
print(f"  {len(high_timeout)} participants exceed threshold")
for pid, rate in high_timeout.items():
    nick = p.loc[p["participantId"] == pid, "nickname"].values
    nick = nick[0] if len(nick) > 0 else "unknown"
    age  = p.loc[p["participantId"] == pid, "age"].values
    age  = age[0] if len(age) > 0 else "?"
    print(f"    {nick} (age {age}): {rate:.1%} timeouts — excluded")
    exclusion_log.append({
        "participantId": pid,
        "nickname":      nick,
        "reason":        f"High timeout rate: {rate:.1%}"
    })

exclude_high_timeout = set(high_timeout.index)

# ── Rule 4: Flag incomplete sessions ─────────────────────────────
# Participants with fewer than 32 main trials (80% of 40) excluded
# Something went wrong with their session
trial_counts = main_t.groupby("participantId").size()
incomplete   = trial_counts[trial_counts < 32]
print(f"\n✓ Rule 4 — Flag participants with fewer than 32 main trials")
print(f"  {len(incomplete)} incomplete sessions found")
for pid, cnt in incomplete.items():
    nick = p.loc[p["participantId"] == pid, "nickname"].values
    nick = nick[0] if len(nick) > 0 else "unknown"
    print(f"    {nick}: only {cnt} trials — excluded")
    exclusion_log.append({
        "participantId": pid,
        "nickname":      nick,
        "reason":        f"Incomplete session: {cnt} trials"
    })

exclude_incomplete = set(incomplete.index)

# ── Rule 5: Flag zero choice variation ───────────────────────────
# If someone chose only A or only B throughout the whole experiment
# they were not engaging with the task — just holding down a key
choice_var = (
    main_t[main_t["timedOut"] == False]
    .groupby("participantId")["choice"]
    .nunique()
)
no_var = choice_var[choice_var < 2]
print(f"\n✓ Rule 5 — Flag participants with zero choice variation")
print(f"  {len(no_var)} participants showed no variation — excluded")
for pid in no_var.index:
    nick = p.loc[p["participantId"] == pid, "nickname"].values
    nick = nick[0] if len(nick) > 0 else "unknown"
    exclusion_log.append({
        "participantId": pid,
        "nickname":      nick,
        "reason":        "Zero choice variation"
    })

exclude_no_var = set(no_var.index)

# ── Apply all exclusions ──────────────────────────────────────────
all_excluded = exclude_high_timeout | exclude_incomplete | exclude_no_var

clean_p = p[~p["participantId"].isin(all_excluded)].copy()
clean_t = main_t[
    (~main_t["participantId"].isin(all_excluded)) &
    (~main_t["is_too_fast"])
].copy()
clean_t = clean_t.drop(columns=["is_too_fast"])

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("EXCLUSION SUMMARY")
print(f"{'=' * 60}")
print(f"  High timeout rate:     {len(exclude_high_timeout)}")
print(f"  Incomplete sessions:   {len(exclude_incomplete)}")
print(f"  Zero choice variation: {len(exclude_no_var)}")
print(f"  Total excluded:        {len(all_excluded)} participants")

print(f"\n{'=' * 60}")
print("CLEAN DATASET SUMMARY")
print(f"{'=' * 60}")
print(f"  Participants: {len(p)} → {len(clean_p)} (removed {len(p) - len(clean_p)})")
print(f"  Trial rows:   {len(main_t)} → {len(clean_t)} (removed {len(main_t) - len(clean_t)})")
print(f"  Avg trials per participant: {clean_t.groupby('participantId').size().mean():.1f}")
print(f"\n  RT statistics after cleaning:")
print(f"    Min:    {clean_t['responseTimeMs'].min()} ms")
print(f"    Max:    {clean_t['responseTimeMs'].max()} ms")
print(f"    Mean:   {clean_t['responseTimeMs'].mean():.0f} ms")
print(f"    Median: {clean_t['responseTimeMs'].median():.0f} ms")
print(f"\n  Timeout rate: {clean_t['timedOut'].mean():.4f}")
print(f"\n  Age distribution:")
print(clean_p["age"].describe().round(1).to_string())

# ── Save outputs ─────────────────────────────────────────────────
clean_p.to_csv(f"{OUTPUT_DIR}/clean_participants.csv", index=False)
clean_t.to_csv(f"{OUTPUT_DIR}/clean_trials.csv",       index=False)

if exclusion_log:
    pd.DataFrame(exclusion_log).to_csv(
        f"{OUTPUT_DIR}/exclusion_log.csv", index=False
    )

print(f"\n✓ Files saved to '{OUTPUT_DIR}/' folder:")
print(f"  clean_participants.csv")
print(f"  clean_trials.csv")
print(f"  exclusion_log.csv")
print("\n→ Run step2_features.py next")
