from xgboost import XGBRegressor
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper
import numpy as np
import joblib
import os
from typing import Type, Dict


class XGBoostRegressorWrapper:
    def __init__(self, scaler_class: Type = MinMaxScalerWrapper(), **kwargs: Dict) -> None:
        self.params = kwargs
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.model_name = f"xgboost_model"
        self.model = None

    def build(self) -> None:
        if self.params:
            self.model = XGBRegressor()
        else:
            self.model = XGBRegressor(**self.params)

    def fit(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, verbose: int = 0) -> None:
        if self.scaler_X is not None:
            X_train = self.scaler_X.fit_transform_X(X_train)
            y_train = self.scaler_y.fit_transform_y(y_train.reshape(len(y_train), 1)).flatten()
            X_val = self.scaler_X.transform_X(X_val)
            y_val = self.scaler_y.transform_y(y_val.reshape(len(y_val), 1)).flatten()
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=verbose)

    def predict(self, X_test: np.array) -> None:
        if self.scaler_X is not None:
            X_test = self.scaler_X.transform_X(X_test)
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1, 1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred.flatten()

    def save(self, save_path: str = "storage") -> None:
        model_filename = f"{save_path}/{self.model_name}"
        if not os.path.exists(model_filename):
            os.makedirs(model_filename)
        joblib.dump(self.model, model_filename)
