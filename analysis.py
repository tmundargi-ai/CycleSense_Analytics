"""
CycleSense Analytics — Phase II: Time Series Forecasting
CS595: Time Series Analysis and Forecasting | Spring 2026
Thejaswini Mundargi — A20586100

Full analysis pipeline:
  1. Data loading & preparation
  2. Exploratory Data Analysis (EDA)
  3. Modeling (Damped Holt-Winters + SARIMA baseline)
  4. Forecasting with prediction intervals
  5. Evaluation & comparison
  6. Risk & uncertainty analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETUP
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1B4F72',
    'secondary': '#2E86C1',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'light': '#D5E8D4',
    'gray': '#7F8C8D',
    'bg': '#FAFAFA'
}
FIG_DIR = './figures'
import os
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, name, dpi=200):
    fig.savefig(f'{FIG_DIR}/{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================
print("=" * 60)
print("1. DATA LOADING & PREPARATION")
print("=" * 60)

df_hourly = pd.read_csv("data.csv")
df_hourly['datetime'] = pd.to_datetime(df_hourly['dteday']) + pd.to_timedelta(df_hourly['hr'], unit='h')

# De-normalize features for interpretability
df_hourly['temp_actual'] = df_hourly['temp'] * 41        # Celsius
df_hourly['atemp_actual'] = df_hourly['atemp'] * 50      # Celsius (feels like)
df_hourly['hum_actual'] = df_hourly['hum'] * 100         # Percentage
df_hourly['windspeed_actual'] = df_hourly['windspeed'] * 67  # km/h

# Aggregate to daily level (forecasting target)
daily = df_hourly.groupby('dteday').agg({
    'cnt': 'sum',
    'casual': 'sum',
    'registered': 'sum',
    'temp_actual': 'mean',
    'atemp_actual': 'mean',
    'hum_actual': 'mean',
    'windspeed_actual': 'mean',
    'holiday': 'max',
    'workingday': 'max',
    'weathersit': lambda x: x.mode()[0],
    'season': 'first',
    'weekday': 'first'
}).reset_index()

daily['date'] = pd.to_datetime(daily['dteday'])
daily = daily.sort_values('date').reset_index(drop=True)
daily.set_index('date', inplace=True)
daily.index.freq = 'D'

print(f"Hourly records: {len(df_hourly):,}")
print(f"Daily records:  {len(daily):,}")
print(f"Date range:     {daily.index.min().date()} to {daily.index.max().date()}")
print(f"Mean daily rentals: {daily['cnt'].mean():.0f}")
print(f"Std daily rentals:  {daily['cnt'].std():.0f}")
print(f"Min: {daily['cnt'].min()}, Max: {daily['cnt'].max()}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Figure 1: Overall Time Series ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(daily.index, daily['cnt'], color=COLORS['primary'], linewidth=0.8, alpha=0.9)
axes[0].set_ylabel('Total Rentals', fontsize=11)
axes[0].set_title('Daily Bike Rental Demand — Washington D.C. (2011–2012)', fontsize=14, fontweight='bold')
axes[0].axhline(daily['cnt'].mean(), color=COLORS['accent'], linestyle='--', alpha=0.5, label=f'Mean = {daily["cnt"].mean():.0f}')
axes[0].legend(fontsize=9)

axes[1].plot(daily.index, daily['casual'], color=COLORS['warning'], linewidth=0.8, label='Casual')
axes[1].plot(daily.index, daily['registered'], color=COLORS['secondary'], linewidth=0.8, label='Registered')
axes[1].set_ylabel('Rentals by Type', fontsize=11)
axes[1].legend(fontsize=9)

axes[2].plot(daily.index, daily['temp_actual'], color=COLORS['accent'], linewidth=0.8, alpha=0.7, label='Temperature (°C)')
ax2 = axes[2].twinx()
ax2.plot(daily.index, daily['hum_actual'], color=COLORS['secondary'], linewidth=0.8, alpha=0.5, label='Humidity (%)')
axes[2].set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_ylabel('Humidity (%)', fontsize=11)
axes[2].set_xlabel('Date', fontsize=11)
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.tight_layout()
save_fig(fig, 'fig1_timeseries_overview')

# --- Figure 2: Seasonal Decomposition ---
decomp = seasonal_decompose(daily['cnt'], model='additive', period=7)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
components = [('Observed', decomp.observed), ('Trend', decomp.trend),
              ('Seasonal (Weekly)', decomp.seasonal), ('Residual', decomp.resid)]
colors_decomp = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['gray']]
for ax, (title, data), c in zip(axes, components, colors_decomp):
    ax.plot(data, color=c, linewidth=0.8)
    ax.set_ylabel(title, fontsize=11)
axes[0].set_title('Seasonal Decomposition of Daily Bike Rentals (Period = 7 days)', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'fig2_seasonal_decomposition')

# --- Figure 3: Hourly Patterns ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By hour of day
hourly_avg = df_hourly.groupby('hr')['cnt'].mean()
working = df_hourly[df_hourly['workingday'] == 1].groupby('hr')['cnt'].mean()
weekend = df_hourly[df_hourly['workingday'] == 0].groupby('hr')['cnt'].mean()
axes[0].plot(working.index, working.values, color=COLORS['primary'], linewidth=2, marker='o', markersize=4, label='Working Day')
axes[0].plot(weekend.index, weekend.values, color=COLORS['warning'], linewidth=2, marker='s', markersize=4, label='Weekend/Holiday')
axes[0].set_xlabel('Hour of Day', fontsize=11)
axes[0].set_ylabel('Average Rentals', fontsize=11)
axes[0].set_title('Hourly Demand Pattern', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_xticks(range(0, 24, 2))

# By day of week
dow_avg = daily.groupby('weekday')['cnt'].agg(['mean', 'std'])
dow_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
bars = axes[1].bar(range(7), dow_avg['mean'], yerr=dow_avg['std'], capsize=3,
                   color=[COLORS['warning'] if i in [0, 6] else COLORS['primary'] for i in range(7)],
                   alpha=0.85, edgecolor='white')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_names)
axes[1].set_xlabel('Day of Week', fontsize=11)
axes[1].set_ylabel('Average Daily Rentals', fontsize=11)
axes[1].set_title('Daily Demand by Day of Week', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'fig3_hourly_dow_patterns')

# --- Figure 4: Seasonal & Weather Impact ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# By season
season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
season_data = daily.groupby('season')['cnt'].agg(['mean', 'std'])
season_colors = ['#27AE60', '#F39C12', '#E67E22', '#3498DB']
axes[0].bar([season_names[i] for i in season_data.index], season_data['mean'],
            yerr=season_data['std'], capsize=4, color=season_colors, alpha=0.85, edgecolor='white')
axes[0].set_ylabel('Average Daily Rentals', fontsize=11)
axes[0].set_title('Demand by Season', fontsize=13, fontweight='bold')

# Temp vs demand
scatter = axes[1].scatter(daily['temp_actual'], daily['cnt'], c=daily['season'],
                          cmap='RdYlBu_r', s=15, alpha=0.6)
axes[1].set_xlabel('Temperature (°C)', fontsize=11)
axes[1].set_ylabel('Daily Rentals', fontsize=11)
axes[1].set_title('Temperature vs Demand', fontsize=13, fontweight='bold')

# Weather situation
weather_names = {1: 'Clear', 2: 'Mist/Cloud', 3: 'Light Rain/Snow', 4: 'Heavy Rain'}
weather_data = daily.groupby('weathersit')['cnt'].agg(['mean', 'std'])
weather_colors = ['#2ECC71', '#F1C40F', '#E74C3C', '#8B0000']
labels = [weather_names.get(i, f'W{i}') for i in weather_data.index]
axes[2].bar(labels, weather_data['mean'], yerr=weather_data['std'], capsize=4,
            color=weather_colors[:len(weather_data)], alpha=0.85, edgecolor='white')
axes[2].set_ylabel('Average Daily Rentals', fontsize=11)
axes[2].set_title('Demand by Weather', fontsize=13, fontweight='bold')
axes[2].tick_params(axis='x', rotation=15)
plt.tight_layout()
save_fig(fig, 'fig4_season_weather_impact')

# --- Figure 5: Correlation Heatmap ---
corr_cols = ['cnt', 'casual', 'registered', 'temp_actual', 'atemp_actual', 'hum_actual', 'windspeed_actual', 'holiday', 'workingday']
corr_labels = ['Total', 'Casual', 'Registered', 'Temp', 'Feels Like', 'Humidity', 'Wind', 'Holiday', 'Workday']
corr = daily[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix — Daily Bike Rental Features', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'fig5_correlation_heatmap')

# --- Figure 6: ACF / PACF ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(daily['cnt'], lags=60, ax=axes[0], color=COLORS['primary'])
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
plot_pacf(daily['cnt'], lags=60, ax=axes[1], color=COLORS['primary'], method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'fig6_acf_pacf')

# --- Figure 7: Monthly boxplot ---
fig, ax = plt.subplots(figsize=(14, 5))
daily['month'] = daily.index.month
daily['year'] = daily.index.year
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
bp = daily.boxplot(column='cnt', by='month', ax=ax, showmeans=True,
                   meanprops={'marker': 'D', 'markerfacecolor': COLORS['accent'], 'markersize': 6},
                   flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
ax.set_xticklabels(month_names)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Daily Rentals', fontsize=11)
ax.set_title('Monthly Distribution of Daily Bike Rentals', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
save_fig(fig, 'fig7_monthly_boxplot')

# ============================================================
# 3. DATA PREPARATION FOR MODELING
# ============================================================
print("\n" + "=" * 60)
print("3. DATA PREPARATION FOR MODELING")
print("=" * 60)

# We forecast daily total rentals (cnt)
# Train: all data except last 14 days
# Test: last 14 days (forecast horizon)
FORECAST_HORIZON = 14

train = daily['cnt'][:-FORECAST_HORIZON]
test = daily['cnt'][-FORECAST_HORIZON:]

print(f"Training period: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
print(f"Test period:     {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")
print(f"Train mean: {train.mean():.0f}, Test mean: {test.mean():.0f}")

# ============================================================
# 4. MODELING
# ============================================================
print("\n" + "=" * 60)
print("4. MODELING")
print("=" * 60)

# --- Model 1: Damped Holt-Winters (Primary) ---
print("\n--- Model 1: Damped Holt-Winters (Multiplicative Seasonality) ---")
hw_model = ExponentialSmoothing(
    train,
    trend='add',
    damped_trend=True,
    seasonal='mul',
    seasonal_periods=7,
    initialization_method='estimated'
).fit(optimized=True)

hw_forecast = hw_model.forecast(FORECAST_HORIZON)
hw_fitted = hw_model.fittedvalues

print(f"Smoothing parameters:")
print(f"  alpha (level):    {hw_model.params['smoothing_level']:.4f}")
print(f"  beta (trend):     {hw_model.params['smoothing_trend']:.4f}")
print(f"  gamma (seasonal): {hw_model.params['smoothing_seasonal']:.4f}")
print(f"  phi (damping):    {hw_model.params['damping_trend']:.4f}")

# --- Model 2: SARIMA (Baseline) ---
print("\n--- Model 2: SARIMA(1,1,1)(1,1,1)[7] (Baseline) ---")
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarima_forecast_result = sarima_model.get_forecast(steps=FORECAST_HORIZON)
sarima_forecast = sarima_forecast_result.predicted_mean
sarima_fitted = sarima_model.fittedvalues

print(f"AIC: {sarima_model.aic:.2f}")
print(f"BIC: {sarima_model.bic:.2f}")

# --- Model 3: Combination (Equal Weight Average) ---
print("\n--- Model 3: Combination (HW + SARIMA average) ---")
comb_forecast = (hw_forecast.values + sarima_forecast.values) / 2
comb_forecast = pd.Series(comb_forecast, index=test.index)

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

def evaluate(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}

results = []
results.append(evaluate(test, hw_forecast, 'Damped Holt-Winters'))
results.append(evaluate(test, sarima_forecast, 'SARIMA(1,1,1)(1,1,1)[7]'))
results.append(evaluate(test, comb_forecast, 'Combination (HW+SARIMA)'))

# Naive baseline (repeat last week)
naive_forecast = train[-7:].values.tolist() * 2
naive_forecast = pd.Series(naive_forecast[:FORECAST_HORIZON], index=test.index)
results.append(evaluate(test, naive_forecast, 'Seasonal Naïve'))

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False, float_format='%.2f'))

# ============================================================
# 6. PREDICTION INTERVALS (Bootstrapped)
# ============================================================
print("\n" + "=" * 60)
print("6. PREDICTION INTERVALS")
print("=" * 60)

# Bootstrap prediction intervals for Holt-Winters
n_boot = 1000
residuals = (train - hw_fitted).dropna().values
boot_forecasts = np.zeros((n_boot, FORECAST_HORIZON))

np.random.seed(42)
for i in range(n_boot):
    noise = np.random.choice(residuals, size=FORECAST_HORIZON, replace=True)
    boot_forecasts[i, :] = hw_forecast.values + noise

# Compute intervals
pi_80_lower = np.percentile(boot_forecasts, 10, axis=0)
pi_80_upper = np.percentile(boot_forecasts, 90, axis=0)
pi_95_lower = np.percentile(boot_forecasts, 2.5, axis=0)
pi_95_upper = np.percentile(boot_forecasts, 97.5, axis=0)

# Ensure non-negative
pi_80_lower = np.maximum(pi_80_lower, 0)
pi_95_lower = np.maximum(pi_95_lower, 0)

# SARIMA prediction intervals (built-in)
sarima_ci_95 = sarima_forecast_result.conf_int(alpha=0.05)
sarima_ci_80 = sarima_forecast_result.conf_int(alpha=0.20)

# Coverage
hw_coverage_95 = np.mean((test.values >= pi_95_lower) & (test.values <= pi_95_upper))
hw_coverage_80 = np.mean((test.values >= pi_80_lower) & (test.values <= pi_80_upper))
sarima_coverage_95 = np.mean((test.values >= sarima_ci_95.iloc[:, 0].values) & (test.values <= sarima_ci_95.iloc[:, 1].values))

print(f"Holt-Winters 95% PI coverage: {hw_coverage_95:.1%}")
print(f"Holt-Winters 80% PI coverage: {hw_coverage_80:.1%}")
print(f"SARIMA 95% PI coverage:       {sarima_coverage_95:.1%}")

# Average interval width
hw_width_95 = np.mean(pi_95_upper - pi_95_lower)
sarima_width_95 = np.mean(sarima_ci_95.iloc[:, 1].values - sarima_ci_95.iloc[:, 0].values)
print(f"\nAvg 95% PI width — HW: {hw_width_95:.0f} rentals, SARIMA: {sarima_width_95:.0f} rentals")

# MSIS (Mean Scaled Interval Score)
def msis(actual, lower, upper, alpha, seasonal_period, train_data):
    """Mean Scaled Interval Score"""
    n = len(actual)
    scale = np.mean(np.abs(np.diff(train_data.values[seasonal_period:])))
    if scale == 0:
        scale = 1
    score = np.mean(
        (upper - lower) +
        (2 / alpha) * np.maximum(lower - actual, 0) +
        (2 / alpha) * np.maximum(actual - upper, 0)
    ) / scale
    return score

hw_msis = msis(test.values, pi_95_lower, pi_95_upper, 0.05, 7, train)
sarima_msis = msis(test.values, sarima_ci_95.iloc[:, 0].values, sarima_ci_95.iloc[:, 1].values, 0.05, 7, train)
print(f"MSIS — HW: {hw_msis:.3f}, SARIMA: {sarima_msis:.3f}")

# ============================================================
# 7. FORECAST VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("7. GENERATING FORECAST FIGURES")
print("=" * 60)

# --- Figure 8: Forecast Comparison ---
fig, ax = plt.subplots(figsize=(14, 6))
# Show last 60 days of training + test
plot_start = train.index[-60]
ax.plot(train[plot_start:], color=COLORS['gray'], linewidth=1.2, label='Historical')
ax.plot(test, color='black', linewidth=2, label='Actual', marker='o', markersize=4)
ax.plot(hw_forecast, color=COLORS['primary'], linewidth=2, linestyle='--', label='Damped HW', marker='s', markersize=4)
ax.plot(sarima_forecast, color=COLORS['accent'], linewidth=2, linestyle='-.', label='SARIMA', marker='^', markersize=4)
ax.plot(comb_forecast, color=COLORS['success'], linewidth=2, linestyle=':', label='Combination', marker='D', markersize=4)
ax.axvline(train.index[-1], color='gray', linestyle=':', alpha=0.5)
ax.fill_between(test.index, pi_95_lower, pi_95_upper, alpha=0.1, color=COLORS['primary'], label='95% PI (HW)')
ax.fill_between(test.index, pi_80_lower, pi_80_upper, alpha=0.2, color=COLORS['primary'], label='80% PI (HW)')
ax.set_title('14-Day Bike Rental Demand Forecast — CycleSense Analytics', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Daily Rentals', fontsize=11)
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
save_fig(fig, 'fig8_forecast_comparison')

# --- Figure 9: Individual Model Forecasts with PIs ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# HW
axes[0].plot(test, color='black', linewidth=2, label='Actual', marker='o', markersize=5)
axes[0].plot(hw_forecast, color=COLORS['primary'], linewidth=2, linestyle='--', label='Forecast', marker='s', markersize=4)
axes[0].fill_between(test.index, pi_95_lower, pi_95_upper, alpha=0.1, color=COLORS['primary'])
axes[0].fill_between(test.index, pi_80_lower, pi_80_upper, alpha=0.2, color=COLORS['primary'])
axes[0].set_title('Damped Holt-Winters Forecast', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Daily Rentals', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', rotation=30)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# SARIMA
axes[1].plot(test, color='black', linewidth=2, label='Actual', marker='o', markersize=5)
axes[1].plot(sarima_forecast, color=COLORS['accent'], linewidth=2, linestyle='--', label='Forecast', marker='^', markersize=4)
axes[1].fill_between(test.index, sarima_ci_95.iloc[:, 0].clip(lower=0), sarima_ci_95.iloc[:, 1], alpha=0.1, color=COLORS['accent'])
axes[1].fill_between(test.index, sarima_ci_80.iloc[:, 0].clip(lower=0), sarima_ci_80.iloc[:, 1], alpha=0.2, color=COLORS['accent'])
axes[1].set_title('SARIMA(1,1,1)(1,1,1)[7] Forecast', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Daily Rentals', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', rotation=30)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
save_fig(fig, 'fig9_individual_forecasts_pi')

# --- Figure 10: Residual Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

hw_resid = (train - hw_fitted).dropna()
axes[0, 0].plot(hw_resid, color=COLORS['primary'], linewidth=0.5, alpha=0.7)
axes[0, 0].axhline(0, color='gray', linestyle='--')
axes[0, 0].set_title('HW Residuals Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Residual')

axes[0, 1].hist(hw_resid, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white', density=True)
axes[0, 1].set_title('HW Residual Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residual')

sarima_resid = sarima_model.resid.dropna()
axes[1, 0].plot(sarima_resid, color=COLORS['accent'], linewidth=0.5, alpha=0.7)
axes[1, 0].axhline(0, color='gray', linestyle='--')
axes[1, 0].set_title('SARIMA Residuals Over Time', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Residual')

axes[1, 1].hist(sarima_resid, bins=40, color=COLORS['accent'], alpha=0.7, edgecolor='white', density=True)
axes[1, 1].set_title('SARIMA Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Residual')

plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, 'fig10_residual_analysis')

# --- Figure 11: Error by Day ---
fig, ax = plt.subplots(figsize=(14, 5))
days = test.index.strftime('%b %d\n%a')
x = np.arange(len(test))
width = 0.25

hw_errors = np.abs(test.values - hw_forecast.values)
sarima_errors = np.abs(test.values - sarima_forecast.values)
comb_errors = np.abs(test.values - comb_forecast.values)

ax.bar(x - width, hw_errors, width, label='Damped HW', color=COLORS['primary'], alpha=0.85)
ax.bar(x, sarima_errors, width, label='SARIMA', color=COLORS['accent'], alpha=0.85)
ax.bar(x + width, comb_errors, width, label='Combination', color=COLORS['success'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(days, fontsize=8)
ax.set_ylabel('Absolute Error (Rentals)', fontsize=11)
ax.set_title('Daily Forecast Error Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
save_fig(fig, 'fig11_daily_errors')

# --- Figure 12: Uncertainty Evolution ---
fig, ax = plt.subplots(figsize=(10, 5))
hw_widths_95 = pi_95_upper - pi_95_lower
hw_widths_80 = pi_80_upper - pi_80_lower
sarima_widths_95 = sarima_ci_95.iloc[:, 1].values - sarima_ci_95.iloc[:, 0].values

ax.plot(range(1, FORECAST_HORIZON + 1), hw_widths_95, color=COLORS['primary'], linewidth=2, marker='o', label='HW 95% PI Width')
ax.plot(range(1, FORECAST_HORIZON + 1), hw_widths_80, color=COLORS['primary'], linewidth=2, marker='s', linestyle='--', label='HW 80% PI Width')
ax.plot(range(1, FORECAST_HORIZON + 1), sarima_widths_95, color=COLORS['accent'], linewidth=2, marker='^', label='SARIMA 95% PI Width')
ax.set_xlabel('Forecast Horizon (Days Ahead)', fontsize=11)
ax.set_ylabel('Prediction Interval Width (Rentals)', fontsize=11)
ax.set_title('Forecast Uncertainty Growth Over Horizon', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xticks(range(1, FORECAST_HORIZON + 1))
plt.tight_layout()
save_fig(fig, 'fig12_uncertainty_evolution')

# ============================================================
# 8. ROLLING-ORIGIN CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("8. ROLLING-ORIGIN CROSS-VALIDATION")
print("=" * 60)

n_origins = 10
step_size = 14
cv_results = {'hw': [], 'sarima': [], 'comb': []}

for i in range(n_origins):
    end_train = len(daily) - FORECAST_HORIZON - i * step_size
    if end_train < 365:
        break
    cv_train = daily['cnt'].iloc[:end_train]
    cv_test = daily['cnt'].iloc[end_train:end_train + FORECAST_HORIZON]

    if len(cv_test) < FORECAST_HORIZON:
        continue

    try:
        hw_cv = ExponentialSmoothing(cv_train, trend='add', damped_trend=True,
                                     seasonal='mul', seasonal_periods=7,
                                     initialization_method='estimated').fit(optimized=True)
        hw_cv_fc = hw_cv.forecast(FORECAST_HORIZON)
        cv_results['hw'].append(mean_absolute_error(cv_test, hw_cv_fc))
    except:
        pass

    try:
        sarima_cv = SARIMAX(cv_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_cv_fc = sarima_cv.get_forecast(steps=FORECAST_HORIZON).predicted_mean
        cv_results['sarima'].append(mean_absolute_error(cv_test, sarima_cv_fc))

        comb_cv = (hw_cv_fc.values + sarima_cv_fc.values) / 2
        cv_results['comb'].append(mean_absolute_error(cv_test, comb_cv))
    except:
        pass

    print(f"  CV fold {i+1}/{n_origins} complete")

print(f"\nCross-Validation MAE (mean ± std):")
for model_name, key in [('Damped HW', 'hw'), ('SARIMA', 'sarima'), ('Combination', 'comb')]:
    if cv_results[key]:
        vals = cv_results[key]
        print(f"  {model_name}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

# ============================================================
# 9. RISK ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("9. RISK & BUSINESS ANALYSIS")
print("=" * 60)

# Identify high-risk days (upper PI exceeds capacity thresholds)
capacity_threshold = 7000  # Example: max bikes available
high_risk_days = np.sum(pi_95_upper > capacity_threshold)
print(f"Days where 95% upper PI exceeds {capacity_threshold} bikes: {high_risk_days}/{FORECAST_HORIZON}")

# Weekend vs weekday uncertainty
test_days = pd.DataFrame({
    'date': test.index,
    'actual': test.values,
    'forecast': hw_forecast.values,
    'pi_lower': pi_95_lower,
    'pi_upper': pi_95_upper,
    'pi_width': pi_95_upper - pi_95_lower,
    'is_weekend': test.index.dayofweek.isin([5, 6]).astype(int)
})

weekend_width = test_days[test_days['is_weekend'] == 1]['pi_width'].mean()
weekday_width = test_days[test_days['is_weekend'] == 0]['pi_width'].mean()
print(f"\nAvg PI width — Weekday: {weekday_width:.0f}, Weekend: {weekend_width:.0f}")

# Save summary metrics
summary = {
    'Training Period': f"{train.index.min().date()} to {train.index.max().date()}",
    'Test Period': f"{test.index.min().date()} to {test.index.max().date()}",
    'Forecast Horizon': f"{FORECAST_HORIZON} days",
    'HW MAE': f"{results_df.loc[results_df['Model']=='Damped Holt-Winters', 'MAE'].values[0]:.1f}",
    'HW RMSE': f"{results_df.loc[results_df['Model']=='Damped Holt-Winters', 'RMSE'].values[0]:.1f}",
    'HW sMAPE': f"{results_df.loc[results_df['Model']=='Damped Holt-Winters', 'sMAPE'].values[0]:.2f}%",
    'SARIMA MAE': f"{results_df.loc[results_df['Model']=='SARIMA(1,1,1)(1,1,1)[7]', 'MAE'].values[0]:.1f}",
    'SARIMA RMSE': f"{results_df.loc[results_df['Model']=='SARIMA(1,1,1)(1,1,1)[7]', 'RMSE'].values[0]:.1f}",
    'Comb MAE': f"{results_df.loc[results_df['Model']=='Combination (HW+SARIMA)', 'MAE'].values[0]:.1f}",
    'HW 95% Coverage': f"{hw_coverage_95:.1%}",
    'HW 80% Coverage': f"{hw_coverage_80:.1%}",
}

print("\n--- Summary Metrics ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — All figures saved to /figures/")
print("=" * 60)
