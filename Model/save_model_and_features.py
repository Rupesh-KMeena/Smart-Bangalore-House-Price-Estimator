import pickle
import json

# Save the model
with open("bangalore_home_prices_model.pickle", "wb") as f:
    pickle.dump(model, f)

# Save the feature names
columns = {"data_columns": [col.lower() for col in X.columns]}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
