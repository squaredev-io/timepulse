from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerWrapper:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()

    def fit_transform(self, data):
        return self.min_max_scaler.fit_transform(data)

    def transform(self, data):
        return self.min_max_scaler.transform(data)

    def inverse_transform(self, data):
        return self.min_max_scaler.inverse_transform(data)