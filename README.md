# CycleSense Analytics — Phase II: Bike-Share Demand Forecasting

**CS 595: Time Series Analysis and Forecasting | Spring 2026**
**Thejaswini Mundargi — A20586100**

## Project Overview

This project forecasts daily bike rental demand for a Washington D.C. bike-share operator 14 days ahead, with prediction intervals to support fleet planning decisions. It was completed as Phase II of the CS 595 course project at Illinois Institute of Technology.

## Repository Structure

```
.
├── analysis.py                  # Full analysis pipeline (EDA, modeling, evaluation)
├── data.csv                     # Hourly bike rental dataset (2011-2012)
├── CycleSense_Report_Thejaswini_Mundargi.pdf     # Final project report
├── Phase2_Presentation.pptx     # Phase II presentation slides
└── README.md
```

## Dataset

- **Source:** Capital Bikeshare, Washington D.C. 
- **Period:** January 1, 2011 — December 31, 2012
- **Records:** 17,379 hourly observations, aggregated to 731 daily observations
- **Target variable:** `cnt` (total daily bike rentals)

## Methods

| Model | Role |
|---|---|
| Seasonal Naive | Baseline |
| Damped Holt-Winters (multiplicative seasonality, period=7) | Primary |
| SARIMA(1,1,1)(1,1,1) s=7 | Alternative |
| 50/50 HW + SARIMA average | Combination |

Prediction intervals: residual bootstrap (n=1,000) for Holt-Winters, analytical Gaussian for SARIMA.

## Key Results (14-day holdout)

| Model | MAE | sMAPE | vs. Naive |
|---|---|---|---|
| Damped Holt-Winters | 818 | 16.2% | -29.2% |
| SARIMA | 843 | 16.9% | -27.0% |
| Combination | 824 | 16.4% | -28.7% |
| Seasonal Naive | 1,155 | 27.6% | — |

## How to Run

### Requirements

- Python 3.8+
- Required packages:

```
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### Steps

1. Clone this repository:
```bash
git clone https://github.com/tmundargi-ai/CycleSense_Analytics.git
cd CycleSense_Analytics
```

2. Make sure `data.csv` is in the project root directory.

3. Open `analysis.py` and update the following two paths near the top of the file:


4. Run the analysis:
```bash
python analysis.py
```

5. All 13 figures will be saved to the `figures/` directory, and all metrics will be printed to the console.

### Expected Output

The script runs the full pipeline and prints results for each section:
- Data summary statistics
- EDA observations
- Model parameters (HW smoothing values, SARIMA AIC/BIC)
- Holdout accuracy (MAE, RMSE, sMAPE for all models)
- Prediction interval coverage and MSIS scores
- Rolling-origin cross-validation results (10 folds)

## References

[1] Makridakis et al., "The M4 Competition," *Int. Journal of Forecasting*, 2020.
[2] Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., 2021.
[3] Fanaee-T & Gama, "Event labeling combining ensemble detectors," *Progress in AI*, 2014.
[4] Hyndman & Koehler, "Another look at measures of forecast accuracy," *Int. Journal of Forecasting*, 2006.
[5] Gardner & McKenzie, "Forecasting trends in time series," *Management Science*, 1985.
[6] Box, Jenkins, Reinsel & Ljung, *Time Series Analysis*, 5th ed., Wiley, 2015.
