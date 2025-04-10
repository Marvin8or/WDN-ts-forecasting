# Imports
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from wdnforecasting.plotting import (
    plot_node_lag,
    plot_node_timeseries,
    plot_node_ACF,
    plot_node_PACF,
)
from wdnforecasting.stationarity import adf_test_statistics
from wdnforecasting.models import (
    PersistanceModel,
    SklearnAutoregression,
    SklearnLassoAutoregression,
)

# %% Simulation setups and simulation run

# Create a water network model
inp_file = "networks/Net1.inp"
wn = wntr.network.WaterNetworkModel(inp_file)
timestep = 10 * 60
duration_seconds = 2 * 24 * 3600
wn.options.time.duration = duration_seconds

wn.options.time.hydraulic_timestep = timestep
wn.options.time.pattern_timestep = timestep
wn.options.time.report_timestep = timestep

# Simulate hydraulics
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Disable controls
if True:
    ctrls_to_disable = wn.control_name_list
    for name in ctrls_to_disable:
        wn.remove_control(name)

# Plot results on the network
if False:
    pressure_at_5hr = results.node["pressure"].loc[5 * 3600, :]
    wntr.graphics.plot_network(
        wn,
        node_attribute=pressure_at_5hr,
        node_size=30,
        title="Pressure at 5 hours",
        node_labels=True,
        directed=True,
    )

# %% Plot time series at node
node_id = "22"
plot_node_lag(
    results,
    quantity="pressure",
    node_id=node_id,
)
plot_node_timeseries(results, node_id)

corr_df = pd.concat(
    [
        results.node["pressure"][node_id].shift(2),
        results.node["pressure"][node_id],
    ],
    axis=1,
)
corr_df.columns = ["t-2", "t"]
corr_results = corr_df.corr()
# Correlation between normal and lagged series
print(corr_results)

# %% Check stationarity

adf_results = adfuller(
    results.node["pressure"][node_id],
    regression="c",
    maxlag=90,
)
adf_test_statistics(adf_results, "Raw pressure data")
print()

diff_pressure_data = np.array(
    [0.0, *np.diff(results.node["pressure"][node_id])]
)
adf_results = adfuller(
    diff_pressure_data,
    regression="c",
    maxlag=90,
)
adf_test_statistics(adf_results, "Diff pressure data")

# %% Plot ACF and PACF functions

acf_lags = 150
plot_node_ACF(
    diff_pressure_data,
    num_lags=acf_lags,
    data_info="diff pressure data",
    xlabel="10 minute lags",
    node_id=node_id,
)

pacf_lags = 100
plot_node_PACF(
    diff_pressure_data,
    num_lags=pacf_lags,
    data_info="diff pressure data",
    xlabel="10 minute lags",
    node_id=node_id,
)


# %% Plot original time series with transformations

time_steps = np.arange(0, duration_seconds + timestep, timestep) / 600

original_pressure_data = results.node["pressure"][node_id]
# seasonal_pressure_data = seasonal_modeling(
#     results.node["pressure"][node_id], 1, 0.02
# )

# diff_seasonal_pressure_data = np.array([0.0, *np.diff(seasonal_pressure_data)])
diff_pressure_data = np.array(
    [0.0, *np.diff(results.node["pressure"][node_id])]
)
diff_diff_pressure_data = np.array([0.0, *np.diff(diff_pressure_data)])
log_pressure_data = np.log(original_pressure_data)

data_transformations = {
    "original pressure data": {
        "data": original_pressure_data,
        "color": "blue",
    },
    "diff pressure data": {"data": diff_pressure_data, "color": "green"},
    "2nd deg diff pressure data": {
        "data": diff_diff_pressure_data,
        "color": "red",
    },
    "log pressure data": {"data": log_pressure_data, "color": "purple"},
}


fig, axs = plt.subplots(4, 1, sharex=True)
fig.suptitle(
    f"Different transformations of raw pressure data\nNode ID: {node_id}"
)
for ri, (tlabel, dtransform) in enumerate(data_transformations.items()):
    axs[ri].plot(
        time_steps, dtransform["data"], label=tlabel, color=dtransform["color"]
    )
    axs[ri].legend()
    axs[ri].grid()

# %% Train Autoregression model


# This goes in preprocessing module
def calculate_lag_features(X: np.ndarray, lags: int) -> np.ndarray:
    # Create and populate matrix with lagged values

    n = len(X)

    # Create empty matrix shape n-lags x lags to store features
    features = np.full(shape=(n - lags, lags), fill_value=np.nan)

    # Create empty matrix shape n-lags x 1 to store targest
    targets = np.full(shape=(n - lags), fill_value=np.nan)

    for ri in range(n - lags):
        features[ri, :] = X[ri : ri + lags]
        targets[ri] = X[ri + lags]

    return features, targets


# %% Evaluate SklearnAutoregression model
lag = 64
test_cutoff_idx = 193
X = diff_pressure_data

time_steps = np.arange(0, duration_seconds + timestep, timestep) / 600
train_time_steps, test_time_steps = (
    time_steps[:test_cutoff_idx],
    time_steps[test_cutoff_idx:],
)

train, test = X[:test_cutoff_idx], X[test_cutoff_idx:]

X_train, y_train = calculate_lag_features(train, lags=lag)

AR_model = SklearnAutoregression(nlags=lag)
AR_model.train(X_train, y_train)

AR_model_predictions = AR_model.walk_forward_prediction(
    train[-lag:], len(test)
)

MSE_AR = mean_squared_error(test, AR_model_predictions)
RMSE_AR = root_mean_squared_error(test, AR_model_predictions)

print(f"MSE (AR model): {MSE_AR}")
print(f"RMSE (AR model): {RMSE_AR}")

# %% Evaluate SklearnAutoregression model that uses only high correlation lags
# This is related to preprocessing
pacf_results, confint = pacf(
    diff_pressure_data,
    nlags=pacf_lags,
    method="ywm",
    alpha=0.05,
)

pacf_lags_arr = np.array([i for i in range(0, pacf_lags + 1)])
valid_idxs = np.abs(pacf_results) > 0.12
valid_pacf_lags = pacf_lags_arr[valid_idxs]
valid_pacf_lags = valid_pacf_lags[valid_pacf_lags < lag]

X_train, y_train = calculate_lag_features(train, lags=lag)

AR_S_model = SklearnLassoAutoregression(
    nlags=lag, specific_lag_indices=valid_pacf_lags
)
AR_S_model.train(X_train, y_train)

AR_S_model_predictions = AR_S_model.walk_forward_prediction(
    train[-lag:], len(test)
)

MSE_AR_S = mean_squared_error(test, AR_S_model_predictions)
RMSE_AR_S = root_mean_squared_error(test, AR_S_model_predictions)

print(f"MSE (AR S model): {MSE_AR_S}")
print(f"RMSE (AR S model): {RMSE_AR_S}")

# %% Evaluate PersistanceModel model
lag = 1
X_test, y_test = calculate_lag_features(test, lags=lag)

PM_model = PersistanceModel()
PM_model_predictions = PM_model.walk_forward_prediction(X_test, len(y_test))
MSE_PM = mean_squared_error(y_test, PM_model_predictions)
RMSE_PM = root_mean_squared_error(y_test, PM_model_predictions)
print(f"MSE (PM model): {MSE_PM}")
print(f"RMSE (PM model): {RMSE_PM}")

# %% Plot predictions
first_value_in_time_series = original_pressure_data.values[0]

train_copy = train.copy()

train_copy[0] = first_value_in_time_series

test_copy = test.copy()

test_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + test_copy[0]
)

AR_predictions_copy = AR_model_predictions.copy()
AR_predictions_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + AR_predictions_copy[0]
)

AR_S_predictions_copy = AR_S_model_predictions.copy()
AR_S_predictions_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1]
    + AR_S_predictions_copy[0]
)

PM_predictions_copy = PM_model_predictions.copy()
PM_predictions_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + PM_predictions_copy[0]
)

# plot predictions vs expected
fig, axs = plt.subplots(2, 1)
fig.suptitle(f"Predictions of models on pressure data\nNode: {node_id}")

axs[0].plot(train_time_steps, train, color="blue", label="Train data")
axs[0].plot(test_time_steps, test, color="green", label="Test data")

axs[0].plot(
    test_time_steps,
    AR_model_predictions,
    color="red",
    label="AR predicted data",
)

axs[0].plot(
    test_time_steps,
    AR_S_model_predictions,
    color="orange",
    label="AR S predicted data",
)
axs[0].plot(
    test_time_steps[1:],
    PM_model_predictions,
    color="magenta",
    label="PM predicted data",
)

axs[0].set_ylabel("Diff pressure [Pa]")
axs[0].grid()
axs[0].legend()

axs[1].plot(
    train_time_steps, np.cumsum(train_copy), color="blue", label="Train data"
)
axs[1].plot(
    test_time_steps, np.cumsum(test_copy), color="green", label="Test data"
)

axs[1].plot(
    test_time_steps,
    np.cumsum(AR_predictions_copy),
    color="red",
    label="AR predicted data",
)

axs[1].plot(
    test_time_steps,
    np.cumsum(AR_S_predictions_copy),
    color="orange",
    label="AR S predicted data",
)
axs[1].plot(
    test_time_steps[1:],
    np.cumsum(PM_predictions_copy),
    color="magenta",
    label="PM predicted data",
)
axs[1].set_xlabel("seconds")
axs[1].set_ylabel("Pressure [Pa]")
axs[1].grid()
axs[1].legend()
