# Smart Bangalore House Price Estimator

## Project Overview

The **Smart Bangalore House Price Estimator** is a machine learning project designed to predict property prices in Bangalore based on various features such as location, size, and number of bedrooms. This project leverages data preprocessing, feature engineering, and model selection techniques to provide accurate and actionable price estimates.

## Features

- **Data Cleaning**: Handles missing values, outliers, and inconsistent data entries.
- **Feature Engineering**: Converts raw data into meaningful features for improved model performance.
- **Model Training**: Utilizes Linear Regression, Lasso Regression, and Decision Tree algorithms.
- **Model Evaluation**: Assesses model performance using cross-validation and grid search.
- **Prediction**: Provides house price predictions based on user inputs.

## Dataset

The dataset used is `bengaluru_house_prices.csv`, which contains information about house prices in Bangalore. It includes features such as location, size, total square feet, number of bathrooms, and price.

## Data Cleaning and Preparation

1. **Loading the Data**:
    ```python
    import pandas as pd

    df = pd.read_csv("/path/to/bengaluru_house_prices.csv")
    ```

2. **Handling Missing Values**:
    ```python
    df.dropna(inplace=True)
    ```

3. **Feature Engineering**:
    ```python
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    ```

4. **Outlier Removal**:
    ```python
    df = df[df.total_sqft/df.bhk >= 300]
    ```

## Model Training

1. **Data Preparation**:
    ```python
    from sklearn.model_selection import train_test_split

    X = df.drop(["price", "location"], axis=1)
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    ```

2. **Training the Model**:
    ```python
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

3. **Model Evaluation**:
    ```python
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X, y, cv=5)
    ```

## Making Predictions

To predict the price of a house:
```python
import numpy as np

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[loc_index] = 1
    return model.predict([x])[0]
```



## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/smart-bangalore-house-price-estimator.git

2. **Navigate to the Project Directory**:

   ```bash
   cd smart-bangalore-house-price-estimator


3. **Run the Prediction Script**:

   ```bash
   python predict_price.py


 ## Contributing

Feel free to fork the repository and submit pull requests. For any issues or enhancements, please open an issue in the GitHub repository.

