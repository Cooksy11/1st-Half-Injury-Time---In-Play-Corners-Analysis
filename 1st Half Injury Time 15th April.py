# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:27:58 2025

@author: Sukhdeep.Sangha
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# === Load data ===
file_path ="1st Half Inj Time 14th April.csv"
df = pd.read_csv(file_path)

# === Sidebar filters ===
st.sidebar.title("Filter Options")
stake_threshold = st.sidebar.selectbox("Select Stake Factor Threshold", [0.99, 0.3, 0.01])
market_filter = st.sidebar.selectbox("Select Market", ['1st Half - Total Corners', 'Total Corners'])

# === Filter data ===
df = df[(df['CURRENTSF'] <= stake_threshold) & (df['Market Name'] == market_filter)].copy()

# === Classify Over/Under ===
def classify_selection(val):
    if isinstance(val, str):
        if val.startswith("Over"):
            return "Over"
        elif val.startswith("Under"):
            return "Under"
    return None

df['Selection'] = df['RESULTDESCRIPTION'].apply(classify_selection)
df = df[df['Selection'].isin(['Over', 'Under'])]

# === Normalize P/L ===
df['Norm_PL'] = df['GGR'] / df['STAKE']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# === Define X-axis bins ===
df['Magnitude_Bin'] = df['Magnitude of change relative'].round().astype(int)

# === Prepare summary ===
summary = []
for bin_val in sorted(df['Magnitude_Bin'].unique()):
    subset = df[df['Magnitude_Bin'] == bin_val]
    if subset.empty:
        continue

    unique_matches = subset['SRC_EVENT_ID'].nunique()
    total_bets = len(subset)
    avg_bets_per_match = total_bets / unique_matches if unique_matches > 0 else 0

    over_bets = subset[subset['Selection'] == 'Over']
    under_bets = subset[subset['Selection'] == 'Under']

    pct_over = 100 * len(over_bets) / total_bets
    avg_over_pl = over_bets['Norm_PL'].mean() if not over_bets.empty else None
    avg_under_pl = under_bets['Norm_PL'].mean() if not under_bets.empty else None

    summary.append({
        'Bin': bin_val,
        'Total_Bets': total_bets,
        'Unique_Matches': unique_matches,
        'Avg_Bets_Per_Match': avg_bets_per_match,
        'Freq_Over': len(over_bets),
        'Freq_Under': len(under_bets),
        'Pct_Over': pct_over,
        'Avg_Over_PL': avg_over_pl,
        'Avg_Under_PL': avg_under_pl
    })

summary_df = pd.DataFrame(summary)

# === Plot ===
x = np.arange(len(summary_df))
fig, ax1 = plt.subplots(figsize=(16, 8))

bar_over = ax1.bar(x, summary_df['Freq_Over'], color='blue', edgecolor='black', label='Over')
bar_under = ax1.bar(x, summary_df['Freq_Under'], bottom=summary_df['Freq_Over'],
                    color='orange', edgecolor='black', label='Under')

ax1.set_ylabel("Frequency of Bets")
ax1.set_ylim(0, 350)
ax1.set_yticks(np.arange(0, 351, 20))
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df['Bin'], fontsize=10)
ax1.set_xlabel("Magnitude of 1st Half Injury Time Change")
ax1.set_title(f"Market: [{market_filter}] | Stake Factor ≤ {stake_threshold}\n"
              "Over/Under Frequency + Normalized P/L + % Over")

for i, row in summary_df.iterrows():
    top = row['Freq_Over'] + row['Freq_Under']
    y_offset = 4
    if pd.notna(row['Avg_Over_PL']):
        color_over = 'green' if row['Avg_Over_PL'] >= 0 else 'red'
        ax1.text(x[i], top + y_offset * 8, f"Over PL: {row['Avg_Over_PL']:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_over)
    if pd.notna(row['Avg_Under_PL']):
        color_under = 'green' if row['Avg_Under_PL'] >= 0 else 'red'
        ax1.text(x[i], top + y_offset * 10, f"Under PL: {row['Avg_Under_PL']:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_under)

    ax1.text(x[i], top + y_offset, f"{row['Unique_Matches']} matches\n{row['Avg_Bets_Per_Match']:.1f} bets/match",
             ha='center', va='bottom', fontsize=8, color='black')

ax2 = ax1.twinx()
ax2.plot(x, summary_df['Pct_Over'], color='red', marker='o', linewidth=2, label='% Over')
ax2.axhline(50, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel("% Over Selections", color='red')
ax2.set_ylim(0, 100)
ax2.set_yticks(np.arange(0, 101, 10))
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.annotate(
    "Legend:\n"
    "↑ Blue = Over Bets | ↓ Orange = Under Bets\n"
    "Red Dot = % of Over Bets | Gray Line = 50%\n\n"
    "Top of towers:\n"
    "- Over PL = Avg Profit/Loss per Over bet (GGR ÷ Stake)\n"
    "- Under PL = Avg Profit/Loss per Under bet\n"
    "Green = Profit | Red = Loss",
    xy=(0.72, 0.62), xycoords='axes fraction',
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray")
)

ax1.grid(axis='y', linestyle='--', alpha=0.6)
st.pyplot(fig)
