# -*- coding: utf-8 -*-
"""
Common methods used to check stationarity of time series
Created on Wed Apr  9 15:46:38 2025

@author: gabri
"""

from typing import Tuple, Dict

ADFResults = Tuple[float, float, int, int, Dict[str, float]]
KPSSResults = Tuple[float, float, int, Dict[str, float]]


def adf_test_statistics(adf_results: ADFResults, info: str) -> None:
    """Prints Augmented Dickey-Fuller Test results"""

    print(f"Augmented Dickey-Fuller Test Results - {info}:")
    print(f"ADF Statistic: {adf_results[0]:.4f}")
    print(f"p-value: {adf_results[1]:.4f}")
    print(f"Lags: {adf_results[2]}")
    print(f"Observations: {adf_results[3]}")
    print("Critical Values:")
    for key, value in adf_results[4].items():
        print(f"  {key}: {value:.4f}")


def kpss_test_statistics(kpss_results: KPSSResults, info: str) -> None:
    """Prints Kwiatowski-Phillips-Scmidt-Shin Test results"""

    print(f"Kwiatowski-Phillips-Scmidt-Shin Test Results - {info}:")
    print(f"KPSS Statistic: {kpss_results[0]:.4f}")
    print(f"p-value: {kpss_results[1]:.4f}")
    print(f"Lags: {kpss_results[2]}")
    print("Critical Values:")
    for key, value in kpss_results[3].items():
        print(f"  {key}: {value:.4f}")
