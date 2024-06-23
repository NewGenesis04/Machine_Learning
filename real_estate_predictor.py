import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from statistics import mean

# Load the dataset
home_data = pd.read_csv("train.csv")

# Print summary statistics and information
# print(home_data.head())
# print(home_data.info())
# print(home_data.describe())

# Count the number of rows with null entries
# num_null_rows = home_data.isnull().sum(axis=0)
# print(num_null_rows)

# # Total number of rows with null entries
# total_null_rows = num_null_rows.sum()
# print("Total number of rows with null entries:", total_null_rows)

# Drop rows with missing values
home_data = home_data.fillna(home_data.mean,axis=0)

# Show the columns of the dataset
print(home_data.columns)

# Target variable
y = home_data.SalePrice
price_mean=mean(home_data.SalePrice)

# Create the list of features
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Define the model
iowa_model = tree.DecisionTreeRegressor(random_state=1)

# Split the dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.6, random_state=1)

# Fit the model
iowa_model.fit(train_X, train_y)

# Make predictions
predictions = iowa_model.predict(test_X)
pred = pd.Series(predictions)
print(pred[:5])
print(test_y[:5])

def get_mae(max_nodes, train_x, val_x, train_y, val_y):
    model = tree.DecisionTreeRegressor(max_leaf_nodes = max_nodes, random_state=1)
    model.fit(train_x, train_y)
    predict = model.predict(val_x)
    mae = mean_absolute_error(val_y, predict)
    return(mae)


candidate = []
for max_nodes in [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                    63, 64, 65, 66, 67, 68, 69, 70, 71]:
    my_mae = get_mae(max_nodes, train_X, test_X, train_y, test_y)
    print(f"Max Leaf Nodes: {max_nodes} \t\t\t MAE: {my_mae}")
    candidate.append((max_nodes, my_mae))

print(candidate)
best_tree_size = min(candidate, key = lambda x: x[1])
print(best_tree_size)

# Fit the final model using the best tree size
final_model = tree.DecisionTreeRegressor(max_leaf_nodes=best_tree_size[0], random_state=1)
final_model.fit(train_X, train_y)

# Make predictions with the final model
final_prediction = final_model.predict(test_X)
final_pred = pd.Series(final_prediction)
print(final_pred[:6])

# Calculate and print the final MAE
final_mae = mean_absolute_error(test_y, final_prediction)
print(f"Mean Absolute Error for the final model: {final_mae} \t\t\t Accuracy: {(final_mae/price_mean)*100} %")





# # Calculate and print Mean Absolute Error (MAE)
# mae = mean_absolute_error(test_y, predictions)
# print(f"Mean Absolute Error: {mae}")



#In-Sample score not good
# print("#########################################################")
# z = mean_absolute_error(y, predictions)
# print(z)

