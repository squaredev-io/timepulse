from sklearn.preprocessing import StandardScaler
import numpy as np

class StandardScalerWrapper:
    def __init__(self) -> None:
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit_transform_X(self, data: np.array) -> np.array:
        return self.scaler_X.fit_transform(data)

    def transform_X(self, data: np.array) -> np.array:
        return self.scaler_X.transform(data)

    def inverse_transform_X(self, data: np.array) -> np.array:
        return self.scaler_X.inverse_transform(data)

    def fit_transform_y(self, data: np.array) -> np.array:
        return self.scaler_y.fit_transform(data)

    def transform_y(self, data: np.array) -> np.array:
        return self.scaler_y.transform(data)

    def inverse_transform_y(self, data: np.array) -> np.array:
        return self.scaler_y.inverse_transform(data)
