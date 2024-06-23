import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from statistics import mean

# Load the dataset
home_data = pd.read_csv("train.csv")

# Drop rows with missing values
home_data = home_data.fillna(home_data.mean,axis=0)

# Show the columns of the dataset
print(home_data.columns)

# Target variable
y = home_data.SalePrice
price_mean= mean(home_data.SalePrice)

# Create the list of features
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

#Split data
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.6, random_state=1)

#Model creation and fitting
model = RandomForestRegressor(random_state = 1)
model.fit(train_X, train_y)

#Predict
pred = pd.Series(model.predict(test_X))
print(pred[:8])

print(test_y[:8])

mae = mean_absolute_error(test_y, pred)
print("Mean Absolute Error: ", mae)
print(f"Accuracy is: {(mae/price_mean)*100} %")