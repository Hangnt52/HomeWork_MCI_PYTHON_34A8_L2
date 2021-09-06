from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import csv

data = pd.read_csv('C:/Users/HangNT/PycharmProjects/HomeWork_MCI_PYTHON_34A8_L2/Linear_Regression/FuelConsumptionCo2.csv')
df = pd.DataFrame(data)
lbel = df.rename(columns={"CO2EMISSIONS": "LABEL"})
print(lbel)
print(df['CO2EMISSIONS'])
"""
def load_fuel_consumption(*, return_X_y=False):
    with open('C:/Users/HangNT/PycharmProjects/HomeWork_MCI_PYTHON_34A8_L2/Linear_Regression/FuelConsumptionCo2.csv') as f:
        data = csv.reader(f)
        temp = next(data)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data)  # names of features
        feature_names = np.array(temp)
        for i, d in enumerate(data):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

        if return_X_y:
            return data, target

        return dict(data=data, target=target, feature_names=feature_names[:-1])
"""