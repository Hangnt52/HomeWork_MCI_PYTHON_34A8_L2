import pandas as pd

from labeler.onehot_load import onehot_encode
from sklearn.preprocessing import StandardScaler

class FuelConsumptionLabeler:
    """
    An object contains 2 components: Attributes (static) and Methods (dynamic)


    """
    def __init__(self, is_take_column=True):
        if is_take_column:
            data_csv = pd.read_csv(
                'C:/Users/HangNT/PycharmProjects/HomeWork_MCI_PYTHON_34A8_L2/Linear_Regression/FuelConsumptionCo2.csv')
            data_fr = pd.DataFrame(data_csv)
            data = data_fr.rename(columns={"CO2EMISSIONS": "LABEL"})
            data = data.drop('TRANSMISSION', axis=1)

            data1 = onehot_encode(data, ['MAKE'], ['MAKE'])
            data2 = onehot_encode(data1, ['MODEL'], ['MODEL'])
            data3 = onehot_encode(data2, ['VEHICLECLASS'], ['VEHICLECLASS'])
            data4 = onehot_encode(data3, ['FUELTYPE'], ['FUELTYPE'])

            self.feature = data4.drop(columns='LABEL').values
            self.label = data4.LABEL.values

        else:
            self.feature, self.label

    def feature_selection(self):
        pass


if __name__ == '__main__':
    fuel_consumption_label = FuelConsumptionLabeler()
    print("DONE")
    # de-coupling
