# Room Occupancy Estimation: Exploratory Data Analysis (INSE-6450)

Exploratory data analysis of the [UCI Room Occupancy Estimation](https://doi.org/10.24432/C5P605) dataset (10,129 rows × 19 columns). The dataset uses temperature, light, sound, CO₂ and PIR sensor readings to estimate how many people (0–3) are in a room.

## Repository contents

| File | Description |
|---|---|
| `Occupancy_Estimation.csv` | Raw dataset |
| `data_analysis.py` | EDA script |
| `requirements.txt` | Python dependencies |
| `INSE_6450_Milestone_1.pdf` | Milestone 1 report |

## Installation

Requires Python 3.9 or newer.

```bash
# optional: create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# install the dependencies
pip install -r requirements.txt
```

## Running the analysis

Run the script from the project folder, since it reads `Occupancy_Estimation.csv` from the current directory:

```bash
python data_analysis.py
```

Each plot opens in its own window, and the script pauses until you close it.

## Outputs

All files are saved in **the folder you run the script from**, which is normally the project folder:

| Output | Content |
|---|---|
| `eda_outliers.csv` | IQR lower/upper bounds, outlier count and outlier % for each numeric column |
| `whisker_plot_standardized.jpeg` | Box-and-whisker plot of all continuous sensors after z-score scaling |
| `correlation_heatmap.jpeg` | Pearson correlation heatmap of the sensors and `Room_Occupancy_Count` |

Printed to the console only:
- Descriptive statistics for every column
- Start and end timestamps, total duration, typical sampling interval and largest gap
- Features ranked by their correlation with `Room_Occupancy_Count`

Shown only, not saved: the target-class pie chart.

## Correlation findings

### Correlation with the target (`Room_Occupancy_Count`, Pearson r)

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
| 16 | S4_Light | 0.356 |

Light is the strongest predictor of occupancy. S4 is the least informative node.

### Strongly correlated feature pairs

| Pair | r |
|---|---|
| S1_Temp – S3_Temp | 0.949 |
| S3_Temp – S4_Temp | 0.885 |
| S1_Temp – S5_CO2 | 0.867 |
| S1_Temp – S4_Temp | 0.855 |
| S1_Light – S2_Light | 0.842 |
| S3_Temp – S5_CO2 | 0.821 |
| S1_Light – S3_Light | 0.816 |

The temperature sensors largely duplicate each other, and so do the light sensors. This suggests averaging each group into one room-wide feature, or dropping the redundant channels.

## Reference

A. P. Singh and S. Chaudhari, "Room Occupancy Estimation," *UCI Machine Learning Repository*, 2018. https://doi.org/10.24432/C5P605
