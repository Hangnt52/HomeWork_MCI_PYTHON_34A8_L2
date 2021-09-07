import pandas as pd
from labeler.linear_regression_label_1 import FuelConsumptionLabeler
from data_loader.fuel_consumption_data_loader import split_train_test_data
from modeler.fuel_comsumption_linear_regression import FuelConsumptionRegression

fuel_consumption_label = FuelConsumptionLabeler()
X_train, X_test, y_train, y_test = split_train_test_data(fuel_consumption_label.feature, fuel_consumption_label.label)

fuel_consumption_model = FuelConsumptionRegression(X_train, X_test, y_train, y_test)
fuel_consumption_model.train()

predicted = fuel_consumption_model.predict_label()

mse, mae, r2_score = fuel_consumption_model.evaluate(predicted)

print(f"Mean squared err: {mse} - Mean absolute error: {mae} - R2 score: {r2_score}")

result = pd.DataFrame({'actual': y_test, 'predicted': predicted})
print(result)

