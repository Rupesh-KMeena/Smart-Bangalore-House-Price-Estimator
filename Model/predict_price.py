import pickle
import json
import numpy as np

# Load the model
with open("bangalore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load the feature names
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

# Function to predict house price
def predict_price(location, sqft, bath, bhk):
    loc_index = columns.index(location.lower()) if location.lower() in columns else -1
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index != -1:
        x[loc_index] = 1
    return model.predict([x])[0]

# Example usage
if __name__ == "__main__":
    location = input("Enter the location: ")
    sqft = float(input("Enter the total square feet: "))
    bath = int(input("Enter the number of bathrooms: "))
    bhk = int(input("Enter the number of bedrooms (BHK): "))
    
    price = predict_price(location, sqft, bath, bhk)
    print(f"The predicted price for the house is: {price:.2f}")
