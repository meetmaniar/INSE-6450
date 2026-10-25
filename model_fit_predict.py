import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, r2_score
from xgboost import XGBRegressor
import os
import time
import platform
from codecarbon import EmissionsTracker

# 1. Load the dataset
df = pd.read_csv('Occupancy_Estimation.csv')

# 2. Combine Date and Time to ensure strict chronological sorting
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)

# 3. Inspect dataset structure and missing values
print("Dataset Shape:", df.shape)
print("Total Missing Values:", df.isnull().sum().sum())

# 4. Inspect target distribution
print("\nTarget Class Distribution (%):")
print(df['Room_Occupancy_Count'].value_counts(normalize=True) * 100)


# 1. Feature Engineering: Compute mean features and drop raw individual sensor columns
df['T_mean'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)
df['L_mean'] = df[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].mean(axis=1)
df['S_mean'] = df[['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']].mean(axis=1)


raw_sensor_cols = [
    'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
    'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
    'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound'
]
df = df.drop(columns=raw_sensor_cols)

# 2. Chronological Data Split (60% Train, 10% Validation, 30% Test)
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.70)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# 3. Print distributions to check temporal skew across splits
print(f"Train set: {len(train_df)} rows | Val set: {len(val_df)} rows | Test set: {len(test_df)} rows\n")

print("--- Train Target Distribution (%) ---")
print(train_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)

print("\n--- Validation Target Distribution (%) ---")
print(val_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)

print("\n--- Test Target Distribution (%) ---")
print(test_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)


# 1. Use 120 smaller contiguous time blocks (~85 rows / 42 mins each)
n_blocks = 120
df['block_id'] = pd.qcut(df.index, q=n_blocks, labels=False)

# 2. Extract maximum occupancy per block for fine-grained stratification
block_summary = df.groupby('block_id')['Room_Occupancy_Count'].agg(
    max_occupancy='max'
).reset_index()

# 3. Stratify blocks into Train (60%) and Temp (40%)
train_blocks, test_val_blocks = train_test_split(
    block_summary['block_id'],
    test_size=0.40,
    random_state=1,
    stratify=block_summary['max_occupancy']
)

# 4. Stratify remaining 40% into Val (10% total) and Test (30% total)
test_val_strat = block_summary.loc[block_summary['block_id'].isin(test_val_blocks), 'max_occupancy']

val_blocks, test_blocks = train_test_split(
    test_val_blocks,
    test_size=0.75,
    random_state=1,
    stratify=test_val_strat
)

# 5. Map blocks back to DataFrames
train_df = df[df['block_id'].isin(train_blocks)].copy().drop(columns=['block_id'])
val_df = df[df['block_id'].isin(val_blocks)].copy().drop(columns=['block_id'])
test_df = df[df['block_id'].isin(test_blocks)].copy().drop(columns=['block_id'])

# 6. Verify distributions
print(f"Train Set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val Set:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test Set:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)\n")

print("--- Train Target Distribution (%) ---")
print(train_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)

print("\n--- Validation Target Distribution (%) ---")
print(val_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)

print("\n--- Test Target Distribution (%) ---")
print(test_df['Room_Occupancy_Count'].value_counts(normalize=True).sort_index() * 100)

# Drop non-feature metadata and target column
drop_cols = ['Date', 'Time', 'Datetime', 'Room_Occupancy_Count']

X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y_train = train_df['Room_Occupancy_Count']

X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
y_val = val_df['Room_Occupancy_Count']

X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test = test_df['Room_Occupancy_Count']


# ==========================================================
# 7. Train an XGBRegressor on the block-stratified split
# ==========================================================
print("\n" + "=" * 60)
print("XGBRegressor: Room_Occupancy_Count (ordinal count regression)")
print("=" * 60)

LABELS = [0, 1, 2, 3]
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    objective="count:poisson",
    eval_metric="mae",
    early_stopping_rounds=30,
    random_state=42,
)


def evaluate(model, X, y):
    """Regression + rounded-classification metrics for one model/split."""
    y_pred_raw = model.predict(X)
    y_pred_rounded = np.clip(np.round(y_pred_raw), 0, 3).astype(int)
    cm = confusion_matrix(y, y_pred_rounded, labels=LABELS)
    recalls = {}
    for i, lbl in enumerate(LABELS):
        support = cm[i].sum()
        recalls[lbl] = cm[i, i] / support if support > 0 else float("nan")
    return {
        "mae": mean_absolute_error(y, y_pred_raw),
        "rmse": mean_squared_error(y, y_pred_raw) ** 0.5,
        "r2": r2_score(y, y_pred_raw),
        "accuracy_rounded": accuracy_score(y, y_pred_rounded),
        "recall_0": recalls[0], "recall_1": recalls[1],
        "recall_2": recalls[2], "recall_3": recalls[3],
        "confusion_matrix": cm,
        "y_pred_raw": y_pred_raw,
        "y_pred_rounded": y_pred_rounded,
    }


def get_peak_memory_mb():
    """Peak resident memory for this process so far, in MB.
    Uses psutil's Windows-specific peak_wset field (falls back to current
    RSS on platforms where peak_wset isn't exposed, e.g. Linux/Mac)."""
    import psutil
    mem = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mem, "peak_wset", None)  # Windows-only field (bytes)
    return (peak if peak is not None else mem.rss) / (1024 ** 2)


# --- Baseline: every training row weighted equally ---
ELECTRICITY_RATE_USD_PER_KWH = 0.15

baseline_model = XGBRegressor(**XGB_PARAMS)
baseline_tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
baseline_tracker.start()
_t0 = time.perf_counter()
baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)
baseline_train_time_s = time.perf_counter() - _t0
baseline_tracker.stop()
baseline_energy_kwh = baseline_tracker.final_emissions_data.energy_consumed
baseline_co2_kg = baseline_tracker.final_emissions_data.emissions
baseline_cost_usd = baseline_energy_kwh * ELECTRICITY_RATE_USD_PER_KWH
print(f"Baseline best iteration: {baseline_model.best_iteration} (of {baseline_model.n_estimators} max)")
print(f"Peak RAM after baseline model: {get_peak_memory_mb():.1f} MB")
baseline_model.save_model("xgb_baseline_model.json")
baseline_model_size_kb = os.path.getsize("xgb_baseline_model.json") / 1024
print(f"Baseline model: {baseline_model.best_iteration} trees x max_depth={XGB_PARAMS['max_depth']}, "
      f"saved size {baseline_model_size_kb:.1f} KB")
print(f"Baseline training energy: {baseline_energy_kwh * 1000:.3f} Wh "
      f"({baseline_co2_kg * 1000:.3f} g CO2eq, est. cost ${baseline_cost_usd:.5f} "
      f"@ ${ELECTRICITY_RATE_USD_PER_KWH}/kWh)")
print(f"Baseline training time: {baseline_train_time_s:.3f} s "
      f"({len(X_train) / baseline_train_time_s:.1f} rows/s)")

# ==========================================================
# 8. Class-imbalance mitigation: sample-weighted XGBRegressor
# ==========================================================
CLASS_WEIGHT = 4.0
sample_weight = np.where(y_train.isin([2, 3]), CLASS_WEIGHT, 1.0)

weighted_model = XGBRegressor(**XGB_PARAMS)
weighted_tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
weighted_tracker.start()
_t0 = time.perf_counter()
weighted_model.fit(
    X_train, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)
weighted_train_time_s = time.perf_counter() - _t0
weighted_tracker.stop()
weighted_energy_kwh = weighted_tracker.final_emissions_data.energy_consumed
weighted_co2_kg = weighted_tracker.final_emissions_data.emissions
weighted_cost_usd = weighted_energy_kwh * ELECTRICITY_RATE_USD_PER_KWH
print(f"Weighted best iteration: {weighted_model.best_iteration} (of {weighted_model.n_estimators} max)")
print(f"Peak RAM after both models: {get_peak_memory_mb():.1f} MB")
weighted_model.save_model("xgb_weighted_model.json")
weighted_model_size_kb = os.path.getsize("xgb_weighted_model.json") / 1024
print(f"Weighted model: {weighted_model.best_iteration} trees x max_depth={XGB_PARAMS['max_depth']}, "
      f"saved size {weighted_model_size_kb:.1f} KB")
print(f"Weighted training energy: {weighted_energy_kwh * 1000:.3f} Wh "
      f"({weighted_co2_kg * 1000:.3f} g CO2eq, est. cost ${weighted_cost_usd:.5f} "
      f"@ ${ELECTRICITY_RATE_USD_PER_KWH}/kWh)")
print(f"Weighted training time: {weighted_train_time_s:.3f} s "
      f"({len(X_train) / weighted_train_time_s:.1f} rows/s)")

# --- Learning curves: training & validation MAE per boosting round ---
baseline_history = baseline_model.evals_result()
weighted_history = weighted_model.evals_result()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(baseline_history["validation_0"]["mae"], linestyle="--", color="#2a78d6", label="Baseline - train")
ax.plot(baseline_history["validation_1"]["mae"], color="#2a78d6", label="Baseline - val")
ax.plot(weighted_history["validation_0"]["mae"], linestyle="--", color="#e34948", label="Weighted - train")
ax.plot(weighted_history["validation_1"]["mae"], color="#e34948", label="Weighted - val")
ax.axvline(baseline_model.best_iteration, color="#2a78d6", linestyle=":", alpha=0.5)
ax.axvline(weighted_model.best_iteration, color="#e34948", linestyle=":", alpha=0.5)
ax.set_xlabel("Boosting round")
ax.set_ylabel("MAE")
ax.set_title("Learning Curves: Baseline vs Class-Weighted XGBRegressor")
ax.legend()
plt.tight_layout()
plt.savefig("xgb_learning_curves.png", dpi=150)
print("\nSaved xgb_learning_curves.png")

# --- Evaluate both models on validation and test, side by side ---
comparison_rows = []
confusion_matrices = {}
for model_name, model in [("baseline", baseline_model), ("weighted", weighted_model)]:
    for split_name, X_split, y_split in [("val", X_val, y_val), ("test", X_test, y_test)]:
        result = evaluate(model, X_split, y_split)
        cm = result.pop("confusion_matrix")
        result.pop("y_pred_raw")
        y_pred_rounded = result.pop("y_pred_rounded")
        confusion_matrices[(model_name, split_name)] = cm
        result["model"] = model_name
        result["split"] = split_name
        comparison_rows.append(result)

        print(f"\n--- {model_name} / {split_name} ---")
        print(f"MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  R2={result['r2']:.4f}  "
              f"Accuracy(rounded)={result['accuracy_rounded']:.4f}")
        print(f"Recall by class: 0={result['recall_0']:.3f}  1={result['recall_1']:.3f}  "
              f"2={result['recall_2']:.3f}  3={result['recall_3']:.3f}")
        cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
        print(cm_df)

# ==========================================================
# 9. Inference efficiency (weighted/final model, on X_test)
# ==========================================================
print("\n" + "=" * 60)
print("Inference Efficiency (weighted/final model)")
print("=" * 60)
print(f"Hardware: {platform.system()} | CPU: {platform.processor() or 'unknown'} | GPU: none (CPU-only)")

# Single-row (batch=1) latency distribution -- the realistic "one prediction
# request at a time" case.
N_LATENCY_SAMPLES = min(200, len(X_test))
single_row_latencies_ms = []
for i in range(N_LATENCY_SAMPLES):
    row = X_test.iloc[[i]]
    _t0 = time.perf_counter()
    weighted_model.predict(row)
    single_row_latencies_ms.append((time.perf_counter() - _t0) * 1000)
single_row_latencies_ms = np.array(single_row_latencies_ms)
p50_latency_ms = np.percentile(single_row_latencies_ms, 50)
p90_latency_ms = np.percentile(single_row_latencies_ms, 90)
print(f"Single-row (batch=1) latency: p50={p50_latency_ms:.3f} ms  p90={p90_latency_ms:.3f} ms  (n={N_LATENCY_SAMPLES})")

# Full-batch latency/throughput -- all of X_test predicted in one call.
_t0 = time.perf_counter()
weighted_model.predict(X_test)
batch_predict_time_s = time.perf_counter() - _t0
throughput_rows_per_s = len(X_test) / batch_predict_time_s
inference_peak_mem_mb = get_peak_memory_mb()
print(f"Batch (batch={len(X_test)}) latency: {batch_predict_time_s * 1000:.2f} ms total "
      f"({(batch_predict_time_s / len(X_test)) * 1000:.4f} ms/row)")
print(f"Throughput: {throughput_rows_per_s:.1f} rows/s")
print(f"Peak RAM so far (process-wide, through inference): {inference_peak_mem_mb:.1f} MB")