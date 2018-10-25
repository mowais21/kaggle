import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melb_data = pd.read_csv('melb_data.csv')

# remove na data
melb_data = melb_data.dropna(axis=0)

# feature to predict
y = melb_data['Price']

# features to use
features = ['Rooms', 'Landsize', 'BuildingArea', 'YearBuilt', 'Bathroom', 'Car', 'Lattitude', 'Longtitude']
X = melb_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# fit the model with data
melb_model = DecisionTreeRegressor()
melb_model.fit(train_X, train_y)

# predict values from the model
val_pred = melb_model.predict(val_X)

# check the mean squared error
print(mean_absolute_error(val_pred, val_y))