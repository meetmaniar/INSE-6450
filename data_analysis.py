import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)

df = pd.read_csv("Occupancy_Estimation.csv")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("datetime").reset_index(drop=True)

print(df.describe(include='all'))
print()
print()
print("******************************************")
# print(df['Room_Occupancy_Count'].value_counts())
df['Room_Occupancy_Count'].value_counts().plot.pie()
plt.title('Target Class')
descriptive_labels = [
    'o',
    '1',
    '2',
    '3'
]
plt.legend(
    title="Number of occupants",
    bbox_to_anchor=(0.85, 1), # Moves legend outside the pie chart
    loc="upper left"
)
plt.show()
plt.savefig("occupancy_count.jpeg", dpi=150)
print("******************************************")
print()
print()

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)

# 2. Calculate time difference between consecutive sensor logs
df['Time_Diff'] = df['Datetime'].diff()

# 3. Print time interval details only
print("******************************************")
print("Start Timestamp:    ", df['Datetime'].min())
print("End Timestamp:      ", df['Datetime'].max())
print("Total Duration:     ", df['Datetime'].max() - df['Datetime'].min())
print("Target Frequency:   ", df['Time_Diff'].mode()[0])
print("Max Interval Gap:   ", df['Time_Diff'].max())
print("******************************************")
print()
print()

COLS = df.columns.tolist()
NUMERIC_COLS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR",
    "Room_Occupancy_Count",
]
outlier_rows = []

for col in NUMERIC_COLS:
    if pd.StringDtype.is_dtype(df[col].dtype):
        continue
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_rows.append({
        "column": col,
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "#outliers": n_outliers, # Outside the normal range
        "%outliers": round(n_outliers / len(df) * 100, 2),
    })
outliers_df = pd.DataFrame(outlier_rows)
outliers_df.to_csv("eda_outliers.csv", index=False)


# Whisker plot #
SCALE_COLS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[SCALE_COLS])
scaled_df = pd.DataFrame(scaled_values, columns=SCALE_COLS)

fig, ax = plt.subplots(figsize=(14, 7))
scaled_df.plot.box(ax=ax, rot=90)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Standardized value (standard deviations from the mean)")
ax.set_title("Standardized Box & Whisker Plot: All Sensors on One Shared Scale")
plt.tight_layout()
plt.savefig("whisker_plot_standardized.jpeg", dpi=150)
print("Saved standardized box & whisker plot to whisker_plot_standardized.jpeg")
plt.show()


# Correlation Heatmap #
NUMERIC_COLS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR",
    "Room_Occupancy_Count",
]
corr = df[NUMERIC_COLS].corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

# Axis ticks + labels
ax.set_xticks(range(len(NUMERIC_COLS)))
ax.set_yticks(range(len(NUMERIC_COLS)))
ax.set_xticklabels(NUMERIC_COLS, rotation=90)
ax.set_yticklabels(NUMERIC_COLS)

for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        value = corr.iloc[i, j]
        text_color = "white" if abs(value) > 0.6 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                 color=text_color, fontsize=7)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson correlation coefficient")

ax.set_title("Correlation Heatmap: Sensor Readings & Occupancy Count")
plt.tight_layout()
plt.savefig("correlation_heatmap.jpeg", dpi=150)
print("Saved correlation heatmap to correlation_heatmap.jpeg")
plt.show()

# --- Call out the strongest relationships with the target ---
target_corr = corr["Room_Occupancy_Count"].drop("Room_Occupancy_Count").sort_values(key=abs, ascending=False)
print()
print("Features most correlated with Room_Occupancy_Count (by |r|):")
print(target_corr.to_string())