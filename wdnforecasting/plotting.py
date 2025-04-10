# -*- coding: utf-8 -*-
"""
Contains common plotting functinalities

Created on Wed Apr  9 15:15:42 2025

@author: gabri
"""
import wntr
import matplotlib
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from typing import Tuple

SimResults = wntr.sim.results.SimulationResults
PltFig = matplotlib.figure.Figure
PltAx = matplotlib.axes.Axes

# Rc params editing
matplotlib.rcParams["font.size"] = 12


def plot_node_lag(
    results: SimResults,
    quantity: str,
    node_id: str,
    lag: int = 1,
):
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"Node: {node_id}\n{quantity.capitalize()} lag {lag}")
    plot_lag(results, ax, quantity, node_id, lag)


def plot_link_lag(
    results: SimResults,
    quantity: str,
    link_id: str,
    lag: int = 1,
):
    fig, ax = plt.subplot(1, 1)
    fig.suptitle(f"Link: {link_id}\n{quantity.capitalize()} lag {lag}")
    plot_lag(results, ax, quantity, link_id, lag)


def plot_lag(
    results: SimResults,
    ax: PltAx,
    quantity: str,
    element_id: str,
    lag: int = 1,
) -> None:
    yt = np.array(results.node[quantity][element_id][:-lag])

    # Get the lagged array
    yt_lag = np.array(results.node[quantity][element_id][lag:])

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


def plot_node_ACF(
    data: np.ndarray, num_lags: int, data_info: str, xlabel: str, node_id: str
) -> None:
    fig, ax = plt.subplots(1, 1)
    title = f"Autocorrelation plot of '{data_info}'\nNode: {node_id}"

    plot_ACF(data, num_lags, title, ax)
    ax.set_xlabel(xlabel)


def plot_link_ACF(
    data: np.ndarray, num_lags: int, data_info: str, xlabel: str, link_id: str
) -> None:
    fig, ax = plt.subplots(1, 1)
    title = f"Autocorrelation plot of '{data_info}'\nLink: {link_id}"

    plot_ACF(data, num_lags, title, ax)
    ax.set_xlabel(xlabel)


def plot_ACF(data: np.ndarray, num_lags: int, title: str, ax: PltAx) -> None:
    sm.graphics.tsa.plot_acf(data, lags=num_lags, title=title, ax=ax)
    ax.set_ylabel("Correlation")
    ax.grid()


def plot_node_PACF(
    data: np.ndarray, num_lags: int, data_info: str, xlabel: str, node_id: str
) -> None:
    fig, ax = plt.subplots(1, 1)
    title = f"Partial Autocorrelation plot of '{data_info}'\nNode: {node_id}"

    plot_PACF(data, num_lags, title, ax)
    ax.set_xlabel(xlabel)


def plot_link_PACF(
    data: np.ndarray, num_lags: int, data_info: str, xlabel: str, link_id: str
) -> None:
    fig, ax = plt.subplots(1, 1)
    title = f"Partial Autocorrelation plot of '{data_info}'\nLink: {link_id}"

    plot_PACF(data, num_lags, title, ax)
    ax.set_xlabel(xlabel)


def plot_PACF(data: np.ndarray, num_lags: int, title: str, ax: PltAx) -> None:
    sm.graphics.tsa.plot_pacf(data, lags=num_lags, title=title, ax=ax)
    ax.set_ylabel("Correlation")
    ax.grid()
