import pandas as pd
import csv

class FuelConsumptionLabeler:
    """
    An object contains 2 components: Attributes (static) and Methods (dynamic)


    """
    def __init__(self, is_take_column=True):
        if is_take_column:
            data = pd.read_csv('C:/Users/HangNT/PycharmProjects/HomeWork_MCI_PYTHON_34A8_L2/Linear_Regression/FuelConsumptionCo2.csv')
            df = pd.DataFrame(data)
            df = pd.DataFrame(data)
            df1 = df.rename(columns={"CO2EMISSIONS": "LABEL"})

            self.feature = df1.drop(columns='LABEL').values
            self.label = df1.LABEL.values

        else:
            self.feature, self.label

    def feature_selection(self):
        pass


if __name__ == '__main__':
    fuel_consumption_label = FuelConsumptionLabeler()
    print("DONE")
    # de-coupling
