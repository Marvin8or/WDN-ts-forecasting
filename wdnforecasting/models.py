# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:52:04 2025

@author: gabri
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class ForecastingModel(ABC):
    """Abstract base class for all forecasting models"""

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on numpy arrays"""
        pass

    @abstractmethod
    def walk_forward_prediction(
        self, X: np.ndarray, nsteps: int
    ) -> np.ndarray:
        """Predict a number of values in the future."""
        pass


class PersistanceModel(ForecastingModel):
    """Persistance model used for benchmarking other models"""

    def __init__(self):
        self._model = lambda x: x

    def train(self, X_train, y_train):
        # Does nothing
        pass

    def walk_forward_prediction(self, X, nsteps):
        predictions = np.zeros(shape=(nsteps,))

        for step in range(nsteps):
            y_hat = self._model(X[step])
            predictions[step] = y_hat

        return predictions


class SklearnAutoregression(ForecastingModel):
    """Autoregression model using scikit-learn LinearRegression method."""

    def __init__(self, nlags: int):
        self._nlags = nlags
        self._model = LinearRegression(fit_intercept=False)

    def train(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def walk_forward_prediction(self, X, nsteps):
        predictions = np.zeros(shape=(nsteps,))

        # A rotating queue that will be updated every time a new prediction comes in
        history_queue = [value for value in X]
        for step in range(nsteps):
            model_input = np.array(history_queue).reshape(1, -1)
            y_hat = self._model.predict(model_input)[0]
            predictions[step] = y_hat

            # Append at the end
            history_queue.append(y_hat)

            # Reshape queue
            history_queue = history_queue[1:]

        return predictions


class SklearnLassoAutoregression(ForecastingModel):
    """Autoregression model using scikit-learn LinearRegression method.
    Contstructor takes additional parameter specific_lag_features
    """

    def __init__(self, nlags: int, specific_lag_indices: np.ndarray):
        self._nlags = nlags
        self._specific_lag_indices = specific_lag_indices
        self._model = LinearRegression(fit_intercept=False)

    def train(self, X_train, y_train):
        self._model.fit(X_train[:, self._specific_lag_indices], y_train)

    def walk_forward_prediction(self, X, nsteps):
        predictions = np.zeros(shape=(nsteps,))

        # A rotating queue that will be updated every time a new prediction comes in
        history_queue = [value for value in X]
        for step in range(nsteps):
            model_input = np.array(history_queue)[
                self._specific_lag_indices
            ].reshape(1, -1)

            y_hat = self._model.predict(model_input)[0]
            predictions[step] = y_hat

            # Append at the end
            history_queue.append(y_hat)

            # Reshape queue
            history_queue = history_queue[1:]

        return predictions
