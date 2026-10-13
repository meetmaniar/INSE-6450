# Room Occupancy Estimation — Exploratory Data Analysis (INSE-6450, Milestone 1)

**Authors:** Manan Mishra · Meet Maniar · Nazmus Sakib Khan (Concordia University)
**Dataset:** [UCI Room Occupancy Estimation](https://doi.org/10.24432/C5P605) (Singh & Chaudhari, 2018), CC BY 4.0

The aim is to estimate **how many people (0–3) are in a room** from non-intrusive sensor readings: temperature, light, sound, CO₂ and PIR motion. The project treats this as a **regression** task, evaluated with MAE, RMSE and R². This repository holds the Milestone 1 work, which is the **exploratory data analysis (EDA)** and data-quality assessment that will guide feature engineering and modelling.

---

## Repository contents

| File | Description |
|---|---|
| `Occupancy_Estimation.csv` | Raw dataset: 10,129 rows × 19 columns |
| `data_analysis.py` | EDA script: summary statistics, target distribution, sampling-interval check, IQR outlier table, standardized box plots, correlation heatmap |
| `INSE_6450_Milestone_1.pdf` | Milestone 1 report: problem statement, data selection, quality issues, feature-engineering plan |
| `requirements.txt` | Python dependencies |

## Setup and running the analysis

### Prerequisites
- Python 3.9 or newer (check with `python --version`)
- `pip` (included with Python)

### 1. Get the project
```bash
git clone https://github.com/meetmaniar/INSE-6450.git
cd INSE-6450
```
(Or open a terminal in the folder that already contains `data_analysis.py`.)

### 2. (Recommended) Create a virtual environment
**Windows (PowerShell / CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the requirements
```bash
pip install -r requirements.txt
```
This installs `pandas`, `numpy`, `matplotlib`, `scikit-learn` and `jinja2`, with the minimum versions listed in `requirements.txt`.

### 4. Run the EDA script
Run it from the project folder, because it reads `Occupancy_Estimation.csv` from the current directory:
```bash
python data_analysis.py
```
The script runs in this order:
1. Prints descriptive statistics for every column.
2. Shows the **target-class pie chart**. Close the window to continue.
3. Prints the start and end timestamps, total duration, typical sampling interval and largest gap.
4. Computes IQR outliers and saves them to `eda_outliers.csv`.
5. Saves and shows the **standardized box-and-whisker plot**. Close it to continue.
6. Saves and shows the **correlation heatmap**, then prints the features ranked by |r| with `Room_Occupancy_Count`.

> **Tip:** each `plt.show()` pauses the script until you close that plot window. To run the script without opening any windows (for example on a server), use the non-interactive backend:
> - macOS/Linux: `MPLBACKEND=Agg python data_analysis.py`
> - Windows PowerShell: `$env:MPLBACKEND="Agg"; python data_analysis.py`
> - Windows CMD: run `set MPLBACKEND=Agg`, then `python data_analysis.py`
>
> The images and CSV are still saved. Only the pop-up windows are skipped.

### Troubleshooting
- `ModuleNotFoundError: No module named 'sklearn'`: the requirements aren't installed in the active environment. Activate the venv and run `pip install -r requirements.txt` again.
- `FileNotFoundError: Occupancy_Estimation.csv`: you are running the script from a different folder. `cd` into the project folder first.

### Outputs

Running the script writes these files to the working directory:

| Output | Content |
|---|---|
| `eda_outliers.csv` | IQR bounds, outlier count and outlier % for each numeric column |
| `whisker_plot_standardized.jpeg` | Box-and-whisker plot of all continuous sensors after z-score scaling |
| `correlation_heatmap.jpeg` | Pearson correlation matrix of the sensors and the target |

It also prints summary statistics, the timestamp and sampling information, and the features ranked by their correlation with the target. Plots open in interactive windows; close each one to continue.

---

## Dataset overview

The data was recorded in a closed room with no HVAC running, using 7 sensor nodes in a star topology. A reading was taken roughly every **30 seconds**.

| Column(s) | Meaning | Unit / type |
|---|---|---|
| `Date`, `Time` | Timestamp, stored as strings and parsed into one `datetime` | — |
| `S1_Temp` … `S4_Temp` | Temperature at 4 nodes | °C |
| `S1_Light` … `S4_Light` | Illuminance at 4 nodes | Lux |
| `S1_Sound` … `S4_Sound` | Sound level (raw analog voltage) | V |
| `S5_CO2` | CO₂ concentration | ppm |
| `S5_CO2_Slope` | Rate of change of CO₂ | ppm / interval |
| `S6_PIR`, `S7_PIR` | Ceiling PIR motion detectors | binary (0/1) |
| `Room_Occupancy_Count` | Ground truth, logged manually | integer 0–3 |

**Data-quality checks:** 0 missing values and 0 duplicate rows (every timestamp is unique).

---

## Key EDA findings

### 1. Time coverage and sampling

| Metric | Value |
|---|---|
| First timestamp | 2017-12-22 10:49:41 |
| Last timestamp | 2018-01-11 09:00:09 |
| Most common interval | 31 s |
| Largest gap | **15 days 06:15** (26 Dec 2017 → 10 Jan 2018, likely the holidays) |
| Other gaps | ~1 day, ~25 min, ~3.5 min |

Records per day:

| Date | Rows | Occupancy present? |
|---|---|---|
| 2017-12-22 | 1,462 | Yes (0–3) |
| 2017-12-23 | 2,779 | Yes (0–3) |
| 2017-12-24 | 1,064 | No, empty all day |
| 2017-12-25 | 1,716 | No, empty all day |
| 2017-12-26 | 1,063 | No, empty all day |
| 2018-01-10 | 997 | Yes (0, 2, 3; no count of 1) |
| 2018-01-11 | 1,048 | No, empty all day |

> **Implication:** All occupied periods fall on 3 of the 7 days. A naive "train on early days, test on later days" split could leave the test set with **no occupied samples, or with only some classes**. The split must be chronological, to avoid leakage between nearby time steps, and must also make sure each split contains occupied periods.

### 2. Target distribution (strong class imbalance)

| Occupants | Rows | Share |
|---|---|---|
| 0 | 8,228 | **81.23 %** |
| 1 | 459 | 4.53 % |
| 2 | 748 | 7.38 % |
| 3 | 694 | 6.85 % |

An empty room makes up most of the data. A model that always predicts 0 would already score a low MAE, so the metrics should also be reported **per occupancy level** (or on occupied samples only).

### 3. How the sensors respond to occupancy (mean reading per class)

| Feature | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| S1_Light (Lux) | 2.7 | 120.2 | 136.1 | 113.3 |
| S2_Light (Lux) | 3.1 | 31.4 | 134.8 | 177.4 |
| S3_Light (Lux) | 13.4 | 54.2 | 138.3 | 156.4 |
| S5_CO2 (ppm) | 404 | 471 | 713 | 851 |
| S5_CO2_Slope | −0.31 | 0.31 | 1.14 | 2.15 |
| S1_Temp (°C) | 25.34 | 25.74 | 26.00 | 26.07 |
| S1_Sound (V) | 0.08 | 0.44 | 0.64 | 0.56 |
| S6_PIR (rate) | 0.00 | 0.34 | 0.48 | 0.54 |
| S7_PIR (rate) | 0.00 | 0.04 | 0.43 | 0.66 |

- **Light** is the clearest signal. An empty room is close to 0 Lux, and S2/S3 keep rising as more people are present (desk lamps at occupied positions).
- **CO₂ and its slope** rise steadily with the number of occupants, which makes them the best features for separating 2 people from 3.
- **Temperature** rises only by fractions of a degree, and it trails the occupancy changes because air takes time to warm.
- **PIR sensors** never fire in an empty room. However, they read 0 in many occupied samples, because a seated person who isn't moving goes undetected.

### 4. Correlation with the target (Pearson r)

| Rank | Feature | r |
|---|---|---|
| 1 | S1_Light | 0.849 |
| 2 | S3_Light | 0.793 |
| 3 | S2_Light | 0.789 |
| 4 | S1_Temp | 0.701 |
| 5 | S7_PIR | 0.695 |
| 6 | S2_Temp | 0.671 |
| 7 | S5_CO2 | 0.660 |
| 8 | S3_Temp | 0.652 |
| 9 | S6_PIR | 0.633 |
| 10 | S5_CO2_Slope | 0.601 |
| 11 | S1_Sound | 0.574 |
| 12 | S2_Sound | 0.558 |
| 13 | S3_Sound | 0.532 |
| 14 | S4_Temp | 0.527 |
| 15 | S4_Sound | 0.460 |
| 16 | S4_Light | **0.356** (weakest) |

**Strongly correlated feature pairs (redundancy):**

| Pair | r |
|---|---|
| S1_Temp – S3_Temp | 0.949 |
| S3_Temp – S4_Temp | 0.885 |
| S1_Temp – S5_CO2 | 0.867 |
| S1_Temp – S4_Temp | 0.855 |
| S1_Light – S2_Light | 0.842 |
| S3_Temp – S5_CO2 | 0.821 |
| S1_Light – S3_Light | 0.816 |

The four temperature sensors carry nearly the same information, and so do the light sensors. This supports collapsing each group into one room-wide aggregate. S4 is the least informative node for light, likely because of where it is placed.

### 5. Outliers (IQR rule, 1.5 × IQR)

| Column | Outliers | % |
|---|---|---|
| S5_CO2_Slope | 4,033 | 39.82 |
| S1_Sound / S2_Sound / S3_Sound | 1,772 / 1,703 / 1,671 | 17.5 / 16.8 / 16.5 |
| S1_Light | 1,716 | 16.94 |
| S5_CO2 | 1,657 | 16.36 |
| S2_Light / S3_Light | 1,017 / 950 | 10.0 / 9.4 |
| S2_Temp | 948 | 9.36 |
| S4_Sound / S4_Light | 721 / 543 | 7.1 / 5.4 |
| S1_Temp | 162 | 1.60 |
| S3_Temp, S4_Temp | 0 | 0.00 |

Most of these "outliers" are **real occupancy events, not errors**. Since 81 % of rows are an empty room, the IQR band fits the empty-room baseline, and every occupied reading falls outside it. (By the same logic the IQR rule flags 18.8 % of the *target* as outliers.) They should **not be removed**. Heavy-tailed variables such as light, sound and CO₂ are candidates for a `log(1 + x)` transform or robust scaling instead.

---

## Implications for modelling (from the EDA)

1. **Splitting:** use a chronological, block-based split (no random shuffling), and check that each split contains occupied periods. Reporting results for each day helps given how few days include occupancy.
2. **Imbalance:** report MAE/RMSE for each occupancy level as well as overall. Consider sample weighting or stratified time blocks.
3. **Feature engineering** (planned in the report):
   - Room-wide means: `T_mean`, `L_mean`, `S_mean` across S1–S4.
   - Keep `S5_CO2` and `S5_CO2_Slope`. They are the strongest signal for telling apart larger occupancy counts.
   - Keep the PIR flags as binary features.
   - Consider lagged or rolling-window features, since CO₂ and temperature trail occupancy changes.
4. **Redundancy:** use the heatmap to drop or aggregate the highly correlated temperature and light channels.
5. **Scaling:** standardize before using distance- or gradient-based models. Apply a log transform to heavy-tailed sensors where it helps the chosen model.

---

## Assumptions

- Readings are taken at regular ~30 s intervals, except for the gaps noted above.
- Sensor positions and the room layout stayed fixed for the whole collection period.
- The room was a closed system (no HVAC or outside influence).
- Ground-truth counts are accurate and synchronized with the sensor readings. A lag in CO₂ and temperature is accepted.

## Notes on `data_analysis.py`

- The `datetime` column is built twice (`datetime` and `Datetime`). One version is enough.
- `descriptive_labels` is defined but never passed to the pie-chart legend. Pass it as `labels=` to show occupant counts in the legend (and correct `'o'` to `'0'`).
- `plt.show()` blocks until each window is closed. Use the `Agg` backend to run the script headless.

## Reference

A. P. Singh and S. Chaudhari, "Room Occupancy Estimation," *UCI Machine Learning Repository*, 2018. https://doi.org/10.24432/C5P605
