"""
=============================================================
Mind vs Clock — Capstone Analysis Pipeline
Step 3: Hidden Markov Model — Strategy Shift Detection
=============================================================
Student:    Anzhelika Simonyan
Supervisor: Arman Asryan
University: American University of Armenia, BS Data Science

WHAT THIS DOES:
    Fits a 2-state Gaussian HMM independently for each participant.
    The two hidden states correspond to:
        State 0 = Analytical strategy (System 2) — slower, deliberate
        State 1 = Intuitive strategy  (System 1) — faster, automatic

    A strategy shift is detected every time the state changes
    from one trial to the next.

    Produces:
        - Per-participant summary (shift count, % intuitive, etc.)
        - Per-trial state labels
        - 10-panel visualization

HOW TO RUN:
    Make sure step2_features.py has been run first.
    Run: python step3_hmm.py

OUTPUT FILES:
    hmm_results.csv           — one row per participant
    hmm_states.csv            — one row per trial with state label
    hmm_visualization.png     — 10-panel results figure
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend (works in PyCharm too)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from hmmlearn import hmm
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette (matches the website theme) ────────────────────
NAVY  = "#0d1b2a"
GOLD  = "#f5c842"
TEAL  = "#2ec4b6"
RED   = "#e84040"
WHITE = "#f4f1eb"
DIMW  = "#8a8680"

# ── Load feature-engineered data ─────────────────────────────────
print("=" * 60)
print("STEP 3 — HIDDEN MARKOV MODEL")
print("=" * 60)

ft = pd.read_csv(f"{OUTPUT_DIR}/featured_trials.csv")
ft = ft.sort_values(["participantId", "trialNumber"]).reset_index(drop=True)

FEATURE_COLS = ["rt_norm", "rt_trend", "choice_entropy",
                "perf_delta_norm", "timeout_flag"]

participants = ft["participantId"].unique()
print(f"\nParticipants to model: {len(participants)}")
print(f"Feature columns:       {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────────
# FIT HMM PER PARTICIPANT
# ─────────────────────────────────────────────────────────────────
results        = []
state_sequences = []
failed         = []

print(f"\nFitting HMMs", end="", flush=True)

for i, pid in enumerate(participants):
    pdata = ft[ft["participantId"] == pid].copy().sort_values("trialNumber")
    X     = pdata[FEATURE_COLS].values

    if len(X) < 10:
        failed.append(pid)
        continue

    # Gaussian HMM with 2 hidden states
    # covariance_type="full" — each state has its own covariance matrix
    # n_iter=200 — max EM iterations
    # random_state=42 — reproducible results
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    try:
        model.fit(X)
        state_seq      = model.predict(X)
        log_likelihood = model.score(X)

        # ── Label states analytically vs intuitively ─────────────
        # The state with HIGHER mean rt_norm = slower = Analytical (System 2)
        # The state with LOWER  mean rt_norm = faster = Intuitive  (System 1)
        mean_rt_s0 = X[state_seq == 0, 0].mean() if (state_seq == 0).any() else 0
        mean_rt_s1 = X[state_seq == 1, 0].mean() if (state_seq == 1).any() else 0

        analytical_state = 0 if mean_rt_s0 >= mean_rt_s1 else 1
        intuitive_state  = 1 - analytical_state

        # Recode: 0 = Analytical, 1 = Intuitive
        labeled_seq = np.where(state_seq == analytical_state, 0, 1)

        # ── Detect shift points ───────────────────────────────────
        # A shift = state changes from trial i-1 to trial i
        shifts = []
        for j in range(1, len(labeled_seq)):
            if labeled_seq[j] != labeled_seq[j - 1]:
                shifts.append({
                    "trial_number":  pdata["trialNumber"].iloc[j],
                    "from_state":    "Analytical" if labeled_seq[j-1] == 0 else "Intuitive",
                    "to_state":      "Analytical" if labeled_seq[j]   == 0 else "Intuitive",
                    "block":         pdata["block"].iloc[j],
                    "timePressure":  pdata["timePressure"].iloc[j],
                })

        n_shifts      = len(shifts)
        pct_intuitive = labeled_seq.mean()

        # Transition probabilities
        tm = model.transmat_
        p_stay_analytical = tm[analytical_state, analytical_state]
        p_stay_intuitive  = tm[intuitive_state,  intuitive_state]

        results.append({
            "participantId":      pid,
            "nickname":           pdata["nickname"].iloc[0],
            "age":                pdata["age"].iloc[0],
            "gender":             pdata["gender"].iloc[0],
            "occupation":         pdata["occupation"].iloc[0],
            "field":              pdata["field"].iloc[0],
            "education":          pdata["education"].iloc[0],
            "stress":             pdata["stress"].iloc[0],
            "sleep":              pdata["sleep"].iloc[0],
            "gaming":             pdata["gaming"].iloc[0],
            "handedness":         pdata["handedness"].iloc[0],
            "n_trials":           len(X),
            "n_shifts":           n_shifts,
            "pct_intuitive":      round(float(pct_intuitive), 4),
            "p_stay_analytical":  round(float(p_stay_analytical), 4),
            "p_stay_intuitive":   round(float(p_stay_intuitive), 4),
            "log_likelihood":     round(float(log_likelihood), 4),
        })

        # Store per-trial state labels
        for j, (_, row) in enumerate(pdata.iterrows()):
            state_sequences.append({
                "participantId":  pid,
                "trialNumber":    row["trialNumber"],
                "block":          row["block"],
                "timePressure":   row["timePressure"],
                "complexity":     row["complexity"],
                "trialType":      row["trialType"],
                "hmm_state":      int(labeled_seq[j]),
                "state_label":    "Intuitive" if labeled_seq[j] == 1 else "Analytical",
                "is_shift":       int(j > 0 and labeled_seq[j] != labeled_seq[j - 1]),
                "rt_norm":        row["rt_norm"],
                "choice_entropy": row["choice_entropy"],
                "responseTimeMs": row["responseTimeMs"],
                "timedOut":       row["timedOut"],
                "pointsEarned":   row["pointsEarned"],
            })

        if (i + 1) % 20 == 0:
            print(".", end="", flush=True)

    except Exception as e:
        failed.append(pid)
        print(f"\n  ⚠ Failed for {pid}: {e}")

print(f" done")

results_df = pd.DataFrame(results)
states_df  = pd.DataFrame(state_sequences)

# ─────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────
print(f"\n✓ HMM fitted for {len(results_df)} / {len(participants)} participants")
if failed:
    print(f"  Failed: {len(failed)}")

print(f"\n{'=' * 60}")
print("HMM RESULTS SUMMARY")
print(f"{'=' * 60}")

print(f"\nStrategy shifts per participant:")
print(results_df["n_shifts"].describe().round(3).to_string())

print(f"\n% time in Intuitive (System 1) state:")
print(results_df["pct_intuitive"].describe().round(3).to_string())

print(f"\nP(stay Analytical):")
print(results_df["p_stay_analytical"].describe().round(3).to_string())

print(f"\nShift distribution:")
print(f"  0 shifts:   {(results_df['n_shifts'] == 0).sum()} participants")
print(f"  1-2 shifts: {((results_df['n_shifts'] >= 1) & (results_df['n_shifts'] <= 2)).sum()} participants")
print(f"  3-5 shifts: {((results_df['n_shifts'] >= 3) & (results_df['n_shifts'] <= 5)).sum()} participants")
print(f"  6+ shifts:  {(results_df['n_shifts'] >= 6).sum()} participants")

shift_rows = states_df[states_df["is_shift"] == 1]
print(f"\nShifts by block:")
print(shift_rows["block"].value_counts().sort_index().to_string())

print(f"\nShifts by time pressure:")
print(shift_rows["timePressure"].value_counts().to_string())

print(f"\nHigh-shifters (≥4): {(results_df['n_shifts'] >= 4).sum()}")
print(f"Low-shifters  (≤1): {(results_df['n_shifts'] <= 1).sum()}")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION — 10-panel figure
# ─────────────────────────────────────────────────────────────────
print(f"\nGenerating visualization...")

plt.style.use("dark_background")
fig = plt.figure(figsize=(20, 24), facecolor=NAVY)
fig.suptitle("Mind vs Clock — HMM Strategy Shift Analysis",
             fontsize=22, color=WHITE, fontweight="bold", y=0.98)

# ── Panel 1: Shift distribution histogram ────────────────────────
ax1 = fig.add_subplot(4, 3, 1)
ax1.set_facecolor(NAVY)
counts = results_df["n_shifts"].value_counts().sort_index()
ax1.bar(counts.index, counts.values, color=GOLD, alpha=0.85, edgecolor=NAVY)
ax1.axvline(results_df["n_shifts"].mean(), color=TEAL, linestyle="--",
            linewidth=1.5, label=f"Mean={results_df['n_shifts'].mean():.1f}")
ax1.set_xlabel("Number of strategy shifts", color=DIMW, fontsize=10)
ax1.set_ylabel("Participants", color=DIMW, fontsize=10)
ax1.set_title("Strategy Shifts Per Participant", color=WHITE, fontsize=12, fontweight="bold")
ax1.tick_params(colors=DIMW)
ax1.legend(fontsize=9, labelcolor=WHITE)
for sp in ax1.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 2: % Intuitive by age group ────────────────────────────
ax2 = fig.add_subplot(4, 3, 2)
ax2.set_facecolor(NAVY)
results_df["age_group"] = pd.cut(results_df["age"],
    bins=[16, 24, 35, 50, 66], labels=["17-24", "25-35", "36-50", "51-65"])
age_means = results_df.groupby("age_group", observed=True)["pct_intuitive"].mean()
bars2 = ax2.bar(age_means.index, age_means.values * 100,
                color=[TEAL, GOLD, RED, "#9b6dff"], alpha=0.85, edgecolor=NAVY)
for bar, val in zip(bars2, age_means.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val * 100:.1f}%", ha="center", va="bottom", color=WHITE, fontsize=9)
ax2.set_xlabel("Age group", color=DIMW, fontsize=10)
ax2.set_ylabel("% Intuitive state", color=DIMW, fontsize=10)
ax2.set_title("Intuitive State by Age Group", color=WHITE, fontsize=12, fontweight="bold")
ax2.tick_params(colors=DIMW)
for sp in ax2.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 3: Shifts by block ──────────────────────────────────────
ax3 = fig.add_subplot(4, 3, 3)
ax3.set_facecolor(NAVY)
block_counts  = shift_rows["block"].value_counts().sort_index()
block_labels  = ["Block 1\n(Low/Low)", "Block 2\n(Low/High)",
                 "Block 3\n(High/Low)", "Block 4\n(High/High)"]
block_colors  = [TEAL, TEAL, RED, RED]
bars3 = ax3.bar(block_labels, block_counts.values, color=block_colors, alpha=0.85, edgecolor=NAVY)
for bar, val in zip(bars3, block_counts.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(val), ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
ax3.set_ylabel("Number of shifts", color=DIMW, fontsize=10)
ax3.set_title("Shifts by Block Condition", color=WHITE, fontsize=12, fontweight="bold")
ax3.tick_params(colors=DIMW)
for sp in ax3.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panels 4-6: State sequence for 3 sample participants ─────────
sample_pids = (
    results_df.nsmallest(1, "n_shifts")["participantId"].tolist() +
    [results_df.iloc[(results_df["n_shifts"] - results_df["n_shifts"].median()).abs().argsort().iloc[0]]["participantId"]] +
    results_df.nlargest(1, "n_shifts")["participantId"].tolist()
)
panel_titles = ["Low-shifter", "Median-shifter", "High-shifter"]

for idx, (pid, ptitle) in enumerate(zip(sample_pids, panel_titles)):
    ax = fig.add_subplot(4, 3, 4 + idx)
    ax.set_facecolor(NAVY)
    pdata = states_df[states_df["participantId"] == pid].sort_values("trialNumber")
    nick  = results_df.loc[results_df["participantId"] == pid, "nickname"].values[0]
    nsh   = results_df.loc[results_df["participantId"] == pid, "n_shifts"].values[0]
    trials = pdata["trialNumber"].values
    states = pdata["hmm_state"].values

    # Shade background by pressure condition
    prev_block = None
    for _, row in pdata.iterrows():
        if row["block"] != prev_block:
            ax.axvspan(row["trialNumber"] - 0.5,
                       trials.max() + 0.5, alpha=0.12,
                       color=RED if row["timePressure"] == "high" else TEAL)
            prev_block = row["block"]

    ax.fill_between(trials, states, alpha=0.4, color=RED,  step="post")
    ax.fill_between(trials, 1 - states, alpha=0.3, color=TEAL, step="post")
    ax.step(trials, states, color=GOLD, linewidth=1.5, where="post")

    for st in pdata[pdata["is_shift"] == 1]["trialNumber"].values:
        ax.axvline(st, color=GOLD, linestyle=":", linewidth=1, alpha=0.8)

    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Analytical", "Intuitive"], color=DIMW, fontsize=8)
    ax.set_xlabel("Trial number", color=DIMW, fontsize=9)
    ax.set_title(f"{ptitle}: {nick}\n{nsh} shifts", color=WHITE, fontsize=11, fontweight="bold")
    ax.tick_params(colors=DIMW)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 7: Shifts by field of study ────────────────────────────
ax7 = fig.add_subplot(4, 3, 7)
ax7.set_facecolor(NAVY)
field_shifts = results_df.groupby("field")["n_shifts"].mean().sort_values()
colors7 = [GOLD if f == "stem" else TEAL for f in field_shifts.index]
ax7.barh(field_shifts.index, field_shifts.values, color=colors7, alpha=0.85, edgecolor=NAVY)
ax7.set_xlabel("Mean strategy shifts", color=DIMW, fontsize=10)
ax7.set_title("Mean Shifts by Field", color=WHITE, fontsize=12, fontweight="bold")
ax7.tick_params(colors=DIMW)
for sp in ax7.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 8: Stress level vs shifts ──────────────────────────────
ax8 = fig.add_subplot(4, 3, 8)
ax8.set_facecolor(NAVY)
stress_shifts = results_df.groupby("stress")["n_shifts"].mean()
ax8.plot(stress_shifts.index, stress_shifts.values,
         color=RED, linewidth=2.5, marker="o", markersize=8, markerfacecolor=GOLD)
ax8.set_xlabel("Stress level (1=calm, 5=stressed)", color=DIMW, fontsize=10)
ax8.set_ylabel("Mean strategy shifts", color=DIMW, fontsize=10)
ax8.set_title("Stress vs Strategy Shifts", color=WHITE, fontsize=12, fontweight="bold")
ax8.set_xticks([1, 2, 3, 4, 5])
ax8.tick_params(colors=DIMW)
for sp in ax8.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 9: Sleep quality vs shifts ─────────────────────────────
ax9 = fig.add_subplot(4, 3, 9)
ax9.set_facecolor(NAVY)
sleep_data = [
    results_df[results_df["sleep"] == "yes"]["n_shifts"].values,
    results_df[results_df["sleep"] == "no"]["n_shifts"].values,
]
bp = ax9.boxplot(sleep_data, tick_labels=["Slept well", "Poor sleep"],
                 patch_artist=True, medianprops={"color": GOLD, "linewidth": 2})
bp["boxes"][0].set_facecolor(TEAL); bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor(RED);  bp["boxes"][1].set_alpha(0.6)
for w in bp["whiskers"]: w.set_color(DIMW)
for c in bp["caps"]:     c.set_color(DIMW)
ax9.set_ylabel("Number of strategy shifts", color=DIMW, fontsize=10)
ax9.set_title("Sleep Quality vs Shifts", color=WHITE, fontsize=12, fontweight="bold")
ax9.tick_params(colors=DIMW)
for sp in ax9.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 10: Shift timing heatmap ───────────────────────────────
ax10 = fig.add_subplot(4, 3, 10)
ax10.set_facecolor(NAVY)
shift_rows2 = states_df[states_df["is_shift"] == 1].copy()
shift_rows2["trial_in_block"] = shift_rows2.groupby(
    ["participantId", "block"]).cumcount() + 1
heatmap_data = shift_rows2.groupby(
    ["block", "trial_in_block"]).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, ax=ax10, cmap="YlOrRd",
            linewidths=0.3, linecolor=NAVY,
            cbar_kws={"label": "Shift count"})
ax10.set_title("Shift Timing Heatmap\n(Block × Trial position)",
               color=WHITE, fontsize=11, fontweight="bold")
ax10.set_xlabel("Trial within block", color=DIMW, fontsize=9)
ax10.set_ylabel("Block", color=DIMW, fontsize=9)
ax10.tick_params(colors=DIMW)

# ── Panel 11: High vs Low shifters RT ────────────────────────────
ax11 = fig.add_subplot(4, 3, 11)
ax11.set_facecolor(NAVY)
results_df["shifter_type"] = results_df["n_shifts"].apply(
    lambda x: "High" if x >= 4 else ("Low" if x <= 1 else None))
rt_means = ft.groupby("participantId")["responseTimeMs"].mean().reset_index()
rt_means.columns = ["participantId", "mean_rt"]
merged   = results_df.dropna(subset=["shifter_type"]).merge(rt_means, on="participantId")
high_rt  = merged[merged["shifter_type"] == "High"]["mean_rt"].values
low_rt   = merged[merged["shifter_type"] == "Low"]["mean_rt"].values
bp2 = ax11.boxplot([high_rt, low_rt],
                   tick_labels=["High-shifters\n(≥4)", "Low-shifters\n(≤1)"],
                   patch_artist=True,
                   medianprops={"color": GOLD, "linewidth": 2})
bp2["boxes"][0].set_facecolor(RED);  bp2["boxes"][0].set_alpha(0.6)
bp2["boxes"][1].set_facecolor(TEAL); bp2["boxes"][1].set_alpha(0.6)
for w in bp2["whiskers"]: w.set_color(DIMW)
for c in bp2["caps"]:     c.set_color(DIMW)
ax11.set_ylabel("Mean RT (ms)", color=DIMW, fontsize=10)
ax11.set_title("RT: High vs Low Shifters", color=WHITE, fontsize=12, fontweight="bold")
ax11.tick_params(colors=DIMW)
for sp in ax11.spines.values(): sp.set_edgecolor("#1e3a52")

# ── Panel 12: Gaming frequency vs shifts ─────────────────────────
ax12 = fig.add_subplot(4, 3, 12)
ax12.set_facecolor(NAVY)
gaming_order  = ["never", "occasionally", "regularly", "daily"]
gaming_shifts = results_df.groupby("gaming")["n_shifts"].mean().reindex(gaming_order)
ax12.bar(gaming_order, gaming_shifts.values,
         color=[DIMW, TEAL, GOLD, RED], alpha=0.85, edgecolor=NAVY)
ax12.set_xlabel("Gaming frequency", color=DIMW, fontsize=10)
ax12.set_ylabel("Mean strategy shifts", color=DIMW, fontsize=10)
ax12.set_title("Gaming Frequency vs Shifts", color=WHITE, fontsize=12, fontweight="bold")
ax12.tick_params(colors=DIMW)
for sp in ax12.spines.values(): sp.set_edgecolor("#1e3a52")

plt.tight_layout(rect=[0, 0, 1, 0.97])
viz_path = f"{OUTPUT_DIR}/hmm_visualization.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight", facecolor=NAVY)
plt.close()
print(f"✓ Visualization saved: {viz_path}")

# ─────────────────────────────────────────────────────────────────
# SAVE RESULT FILES
# ─────────────────────────────────────────────────────────────────
results_df.to_csv(f"{OUTPUT_DIR}/hmm_results.csv", index=False)
states_df.to_csv(f"{OUTPUT_DIR}/hmm_states.csv",   index=False)

print(f"\n✓ Files saved to '{OUTPUT_DIR}/' folder:")
print(f"  hmm_results.csv       — one row per participant")
print(f"  hmm_states.csv        — one row per trial with state label")
print(f"  hmm_visualization.png — 10-panel results figure")
print("\n→ Run step4_regression.py next")
