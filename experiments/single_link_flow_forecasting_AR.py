import wntr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

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
ctrls_to_disable = wn.control_name_list
for name in ctrls_to_disable:
    wn.remove_control(name)

# Plot results on the network
if False:
    pressure_at_5hr = results.node["pressure"].loc[172800, :]
    wntr.graphics.plot_network(
        wn,
        node_attribute=pressure_at_5hr,
        node_size=30,
        title="Pressure at 5 hours for Net1",
        link_labels=True,
        node_labels=True,
        directed=True,
    )

# %% PLot time series at link
from wntr.sim.results import SimulationResults


def plot_link_timeseries(
    results: SimulationResults,
    link_id: str,
    quantites_to_plot: List[str] = None,
    xlabel: str = "timestamps [s]",
) -> None:
    if quantites_to_plot is None:
        quantites_to_plot = results.link.keys()

    # TODO check if all provided quantities are in results.link.keys()
    # If not raise error
    fig, axs = plt.subplots(len(quantites_to_plot), 1, sharex=True)

    fig.suptitle(f"Link: {link_id}")
    for i, k in enumerate(quantites_to_plot):
        axs[i].plot(
            results.link[k].index.values,
            results.link[k][link_id].values,
            # label=k,
        )
        axs[i].plot(
            results.link[k].index.values,
            results.link[k][link_id].values,
            "bo",
            label=k,
        )
        axs[i].set_ylabel(k)
        axs[i].grid()
        axs[i].legend()
    axs[-1].set_xlabel(xlabel)


link_id = "21"
plot_link_timeseries(results, link_id)

# %% Compare all flow rate time series from all Node 22 links
node_id = "22"
quantity = "headloss"
xlabel = "Timesteps [s]"
adjacent_links = wn.get_links_for_node(node_id)

fig, axs = plt.subplots(len(adjacent_links), sharex=True, sharey=True)
fig.suptitle(f"{quantity.capitalize()} [m] of Node: {node_id} adjacent links")

for li, link_name in enumerate(adjacent_links):
    axs[li].plot(
        results.link[quantity].index.values,
        results.link[quantity][link_name].values,
    )
    axs[li].plot(
        results.link[quantity].index.values,
        results.link[quantity][link_name].values,
        "bo",
        label=f"Link: {link_name}",
    )
    axs[li].grid()
    axs[li].legend()
axs[-1].set_xlabel(xlabel)


# %% Check stationarity
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import butter, filtfilt


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


adf_results = adfuller(
    results.link["flowrate"][link_id],
    regression="c",
    maxlag=90,
)
adfuller_statistics(adf_results, "Raw flowrate data")
print()
diff_flowrate_data = np.array(
    [0.0, *np.diff(results.link["flowrate"][link_id])]
)
adf_results = adfuller(
    diff_flowrate_data,
    regression="c",
    maxlag=90,
)
adfuller_statistics(adf_results, "Diff flowrate data")

# %%% Plot ACF and PACF functions
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf

# sm.graphics.tsa.plot_acf(
#     diff_flowrate_data, lags=60, title=f"Autocorrelation\nLink: {link_id}"
# )
# plt.grid()
# plt.xlabel("10 min lags")

pacf_nlags = 100
sm.graphics.tsa.plot_pacf(
    diff_flowrate_data,
    lags=pacf_nlags,
    title=f"Partial Autocorrelation\nLink: {link_id}",
)
plt.grid()
plt.xlabel("10 min lags")

pacf_results, confint = pacf(
    diff_flowrate_data,
    nlags=pacf_nlags,
    method="ywm",
    alpha=0.05,
)
pacf_lags = np.array([i for i in range(1, pacf_nlags + 2)])

valid_idxs = np.abs(pacf_results) > 0.12
pacf_lags = pacf_lags[valid_idxs]
# %% Plot original time series with transformations

time_steps = np.arange(0, duration_seconds + timestep, timestep) / 600

original_flowrate_data = results.link["flowrate"][link_id]

diff_flowrate_data = np.array(
    [0.0, *np.diff(results.link["flowrate"][link_id])]
)
diff_diff_flowrate_data = np.array([0.0, *np.diff(diff_flowrate_data)])
log_flowrate_data = np.log(original_flowrate_data)

data_transformations = {
    "original flowrate data": {
        "data": original_flowrate_data,
        "color": "blue",
    },
    "diff flowrate data": {"data": diff_flowrate_data, "color": "green"},
    "2nd deg diff flowrate data": {
        "data": diff_diff_flowrate_data,
        "color": "red",
    },
    "log flowrate data": {"data": log_flowrate_data, "color": "purple"},
}


fig, axs = plt.subplots(4, 1, sharex=True)
fig.suptitle(
    f"Different transformations of raw flowrate data\nLink ID: {link_id}"
)
for ri, (tlabel, dtransform) in enumerate(data_transformations.items()):
    axs[ri].plot(
        time_steps, dtransform["data"], label=tlabel, color=dtransform["color"]
    )
    axs[ri].legend()
    axs[ri].grid()

# %% Train Autoregression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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


lag = 66  # To be decided with PACF plot
num_retries = 5
time_steps = np.arange(0, duration_seconds + timestep, timestep) / 600

X = diff_flowrate_data

test_cutoff_idx = 193

train, test = X[:test_cutoff_idx], X[test_cutoff_idx:]

train_time_steps, test_time_steps = (
    time_steps[:test_cutoff_idx],
    time_steps[test_cutoff_idx:],
)
X_train, y_train = calculate_lag_features(train, lags=lag)

# X_train, y_train = calculate_specific_lag_features(
#     train, max_lags=lag, spec_lags=pacf_lags
# )

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

        model_input = np.array(history)[:].reshape(1, -1)
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
print(f"MSE (AR model): {MSE_AR}")


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
    print(f"MSE (PM model): {MSE_persistance_model}")
    return persistance_model_predictions, MSE_persistance_model


PM_predictions, MSE_PM = validate_persistance_model(diff_flowrate_data)
# %% Plot predictions
first_value_in_time_series = original_flowrate_data.values[0]

train_copy = train.copy()

train_copy[0] = first_value_in_time_series

test_copy = test.copy()

test_copy[0] = (
    original_flowrate_data.values[test_cutoff_idx - 1] + test_copy[0]
)

AR_predictions_copy = AR_predictions.copy()
AR_predictions_copy[0] = (
    original_flowrate_data.values[test_cutoff_idx - 1] + AR_predictions_copy[0]
)
PM_predictions_copy = PM_predictions.copy()
PM_predictions_copy[0] = (
    original_flowrate_data.values[test_cutoff_idx - 1] + PM_predictions_copy[0]
)
# plot predictions vs expected
fig, axs = plt.subplots(1, 1)
fig.suptitle(f"AR predictions of flowrate data\nLink: {link_id}")

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
axs.set_ylabel("Flowrate [m3/s]")
axs.grid()
axs.legend()
