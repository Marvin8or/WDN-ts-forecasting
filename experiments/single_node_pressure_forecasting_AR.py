# Imports
import wntr
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# %% Simulation setups and simulation run

# Create a water network model
inp_file = "examples/networks/Net1.inp"
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


def plot_lag(
    results: wntr.sim.results.SimulationResults,
    quantity: str,
    node_id: str,
    lag: int = 1,
) -> None:
    yt = np.array(results.node[quantity][node_id][:-lag])

    # Get the lagged array
    yt_lag = np.array(results.node[quantity][node_id][lag:])

    fig, ax = plt.subplots(1, 1)
    ax.plot(yt, yt_lag, "bo")
    ax.set_xlabel("y(t)")
    ax.set_ylabel(f"y(t - {lag})")


def plot_node_timeseries(
    results, node_id: str, xlabel: str = "timestamps [s]"
) -> None:
    fig, axs = plt.subplots(4, 1, sharex=True)

    fig.suptitle(f"Node: {node_id}")
    for i, k in enumerate(results.node.keys()):
        axs[i].plot(
            results.node[k].index.values,
            results.node[k][node_id].values,
            # label=k,
        )
        axs[i].plot(
            results.node[k].index.values,
            results.node[k][node_id].values,
            "bo",
            label=k,
        )
        axs[i].set_ylabel(k)
        axs[i].grid()
        axs[i].legend()
    axs[-1].set_xlabel(xlabel)


node_id = "22"
# plot_lag(results, "pressure", node_id, lag=1)
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


# autocorrelation_plot(results.node["pressure"][node_id])
# plot_acf(results.node["pressure"][node_id], lags=12)

# %% Check stationarity


def adfuller_statistics(adf_results, info: str) -> None:
    """Prints Augmented Dickey-Fuller Test results"""

    print(f"Augmented Dickey-Fuller Test Results - {info}:")
    print(f"ADF Statistic: {adf_results[0]:.4f}")
    print(f"p-value: {adf_results[1]:.4f}")
    print(f"Lags: {adf_results[2]}")
    print(f"Observations: {adf_results[3]}")
    print("Critical Values:")
    for key, value in adf_results[4].items():
        print(f"  {key}: {value:.4f}")


def kpss_statistics(kpss_results, info: str) -> None:
    """Prints Kwiatowski-Phillips-Scmidt-Shin Test results"""

    print(f"Kwiatowski-Phillips-Scmidt-Shin Test Results - {info}:")
    print(f"KPSS Statistic: {kpss_results[0]:.4f}")
    print(f"p-value: {kpss_results[1]:.4f}")
    print(f"Lags: {kpss_results[2]}")
    print("Critical Values:")
    for key, value in kpss_results[3].items():
        print(f"  {key}: {value:.4f}")


# adf_results = adfuller(
#     results.node["pressure"][node_id], regression="c", maxlag=100
# )
# adfuller_statistics(adf_results)
# print()
# kpss_results = kpss(
#     results.node["pressure"][node_id], regression="c", nlags=100
# )
# kpss_statistics(kpss_results)


def seasonal_differencing(
    data: pd.Series, seasonal_period_samples: int, dropna=True
):
    result = data - data.shift(seasonal_period_samples)
    if dropna:
        result = result.dropna()
    return result.values


# def seasonal_modeling(
#     x_data: np.ndarray, y_data: np.ndarray, polydeg: int
# ) -> np.ndarray:
#     """Fits a polinomial to the data to capture the seasoanl component."""

#     poly_coefficients = np.polyfit(x_data, y_data, polydeg)
#     seasonal_component = np.polyval(poly_coefficients, x_data)
#     return seasonal_component


# def seasonal_modeling(y_data: np.ndarray, N: int, Wn: float) -> np.ndarray:
#     b, a = butter(N, Wn)

#     seasonal_component = filtfilt(b, a, y_data)
#     return seasonal_component

diff_pressure_data = np.array(
    [0.0, *np.diff(results.node["pressure"][node_id])]
)

adf_results = adfuller(
    results.node["pressure"][node_id],
    regression="c",
    maxlag=90,
)
adfuller_statistics(adf_results, "Raw pressure data")
print()
adf_results = adfuller(
    diff_pressure_data,
    regression="c",
    maxlag=90,
)
adfuller_statistics(adf_results, "Diff pressure data")
# kpss_results = kpss(
#     results.node["pressure"][node_id],
#     nlags=95,
# )
# kpss_statistics(kpss_results)
# print()
# kpss_results = kpss(
#     diff_pressure_data,
#     nlags=95,
# )
# kpss_statistics(kpss_results)
# %%% Plot ACF and PACF functions


# sm.graphics.tsa.plot_acf(
#     diff_pressure_data, lags=150, title=f"Autocorrelation\nNode: {node_id}"
# )

pacf_nlags = 100
sm.graphics.tsa.plot_pacf(
    diff_pressure_data,
    lags=pacf_nlags,
    title=f"Partial Autocorrelation\nNode: {node_id}",
)
plt.grid()
plt.xlabel("10 min lags")

pacf_results, confint = pacf(
    diff_pressure_data,
    nlags=pacf_nlags,
    method="ywm",
    alpha=0.05,
)
pacf_lags = np.array([i for i in range(1, pacf_nlags + 2)])

valid_idxs = np.abs(pacf_results) > 0.12
pacf_lags = pacf_lags[valid_idxs]


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


def calculate_specific_lag_features(
    X: np.ndarray, max_lags: int, spec_lags: np.ndarray
) -> np.ndarray:
    # Create and populate matrix with specific lagged values
    spec_lag = np.sort(spec_lags)
    spec_lag_idxs = spec_lag - 1

    spec_lag_idxs = spec_lag_idxs[spec_lag < max_lags]
    features, targets = calculate_lag_features(X, max_lags)

    features = features[:, spec_lag_idxs]

    return features, targets


lag = 64  # To be decided with PACF plot
num_retries = 5
time_steps = np.arange(0, duration_seconds + timestep, timestep) / 600

X = diff_pressure_data

test_cutoff_idx = 193

train, test = X[:test_cutoff_idx], X[test_cutoff_idx:]

train_time_steps, test_time_steps = (
    time_steps[:test_cutoff_idx],
    time_steps[test_cutoff_idx:],
)
# X_train, y_train = calculate_lag_features(train, lags=lag)

X_train, y_train = calculate_specific_lag_features(
    train, max_lags=lag, spec_lags=pacf_lags
)

AR_model = LinearRegression(fit_intercept=False)
AR_model.fit(X_train, y_train)

# Walk forward validation
avg_predictions = np.array([])
for retry in range(num_retries):
    predictions = []

    history = [val for val in train[-lag:]]
    for obs in test:
        pacf_lags = np.sort(pacf_lags)
        spec_lag_idxs = pacf_lags - 1

        spec_lag_idxs = spec_lag_idxs[pacf_lags < lag]

        model_input = np.array(history)[spec_lag_idxs].reshape(1, -1)
        y_hat = AR_model.predict(
            model_input  # + np.random.normal(loc=0, scale=np.std(X))
        )
        predictions.append(y_hat[0])

        history = history[1:]
        history.append(y_hat[0])
    if retry == 0:
        avg_predictions = np.array(predictions)
    else:
        avg_predictions += predictions

avg_predictions /= num_retries


AR_predictions = np.array(avg_predictions)
MSE_AR = mean_squared_error(test, predictions)
RMSE_AR = root_mean_squared_error(test, predictions)

print(f"MSE (AR model): {MSE_AR}")
print(f"RMSE (AR model): {RMSE_AR}")

# %% Validate persistance model


def persistance_model(x):
    return x


def validate_persistance_model(data: np.ndarray) -> None:
    # Create the dataset

    lag = 1  # Persistance model only needs value before
    X = data

    test_cutoff_idx = 193

    _, test = X[:test_cutoff_idx], X[test_cutoff_idx:]
    X_test, y_test = calculate_lag_features(test, lags=lag)

    # Walk forward validation
    persistance_model_predictions = []
    for obs in X_test:
        y_hat = persistance_model(obs)
        persistance_model_predictions.append(y_hat)

    persistance_model_predictions = np.array(persistance_model_predictions)
    MSE_persistance_model = mean_squared_error(
        y_test, persistance_model_predictions
    )
    RMSE_persistance_model = root_mean_squared_error(
        y_test, persistance_model_predictions
    )
    print(f"MSE (PM model): {MSE_persistance_model}")
    print(f"RMSE (PM model): {RMSE_persistance_model}")

    return persistance_model_predictions, MSE_persistance_model


PM_predictions, MSE_PM = validate_persistance_model(diff_pressure_data)
# %% Plot predictions
first_value_in_time_series = original_pressure_data.values[0]

train_copy = train.copy()

train_copy[0] = first_value_in_time_series

test_copy = test.copy()

test_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + test_copy[0]
)

AR_predictions_copy = AR_predictions.copy()
AR_predictions_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + AR_predictions_copy[0]
)
PM_predictions_copy = PM_predictions.copy()
PM_predictions_copy[0] = (
    original_pressure_data.values[test_cutoff_idx - 1] + PM_predictions_copy[0]
)
# plot predictions vs expected
fig, axs = plt.subplots(1, 1)
fig.suptitle(f"AR predictions of pressure data\nNode: {node_id}")

axs.plot(
    train_time_steps, np.cumsum(train_copy), color="blue", label="Train data"
)
axs.plot(
    test_time_steps, np.cumsum(test_copy), color="green", label="Test data"
)
axs.plot(
    test_time_steps,
    np.cumsum(AR_predictions_copy),
    color="red",
    label="AR predicted data",
)

axs.plot(
    test_time_steps[1:],
    np.cumsum(PM_predictions_copy),
    color="orange",
    label="PM predicted data",
)
axs.set_xlabel("seconds")
axs.set_ylabel("Pressure [Pa]")
axs.grid()
axs.legend()
