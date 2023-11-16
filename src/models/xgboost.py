from xgboost import XGBRegressor
from src.processing.min_max_scaler import MinMaxScalerWrapper
from datetime import datetime
import joblib
import os

class XGBRegressorModel:
    def __init__(self, scaler_class=MinMaxScalerWrapper(), **kwargs):
        self.params = kwargs
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.model_name = f"xgboost_model_{datetime.now().strftime('D%Y-%m-%dT%H.%M')}"
        self.model = None
        

    def build(self):
        if self.params is None:
            self.model = XGBRegressor()
        else:
            self.model = XGBRegressor(**self.params)


    def fit(self, X_train, y_train, X_val, y_val):
        if self.scaler_X is not None:
            X_train = self.scaler_X.fit_transform_X(X_train)
            y_train = self.scaler_y.fit_transform_y(y_train.reshape(len(y_train), 1)).flatten()
            X_val = self.scaler_X.transform_X(X_val)
            y_val = self.scaler_y.transform_y(y_val.reshape(len(y_val), 1)).flatten()
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        if self.scaler_X is not None:
            X_test = self.scaler_X.fit_transform_X(X_test)
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1,1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred.flatten()
    

    def save(self, save_path='storage'):
        model_filename = f'{save_path}/{self.model_name}'
        if not os.path.exists(model_filename):
            os.makedirs(model_filename)
        joblib.dump(self.model, model_filename)


