import sys
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from datetime import datetime, timedelta
import json


def preprocess_data(data, product_name):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Group data by week and month, and calculate the total sales
    data_monthly = data.groupby(pd.Grouper(key='Date', freq='M')).sum().reset_index()

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    data_monthly['Promotion'] = data_monthly['Promotion'].astype(str)  # Convert to string type
    data_monthly['Promotion'].fillna(data_monthly['Promotion'].mode()[0], inplace=True)
    data_monthly[['Price', 'CompetitorPrice', 'CompetitorCount']] = imputer.fit_transform(
        data_monthly[['Price', 'CompetitorPrice', 'CompetitorCount']])

    # Replace 'Yes' and 'No' values with default values before converting to int
    default_promotion = 0
    promotion_mapping = {'Yes': 1, 'No': 0}
    data_monthly['Promotion'] = data_monthly['Promotion'].map(promotion_mapping).fillna(default_promotion).astype(int)

    # Feature scaling
    scaler = StandardScaler()
    data_monthly[['Price', 'CompetitorPrice', 'CompetitorCount']] = scaler.fit_transform(
        data_monthly[['Price', 'CompetitorPrice', 'CompetitorCount']])

    # Preprocess data based on the product name if needed
    if product_name == 'Banana':
        # Preprocess for Banana
        pass
    elif product_name == 'Bread':
        # Preprocess for Bread
        pass
    elif product_name == 'Milk':
        # Preprocess for Milk
        pass
    elif product_name == 'Orange':
        # Preprocess for Orange
        pass
    elif product_name == 'Sweet':
        # Preprocess for Sweet
        pass

    return {'monthly': data_monthly.drop(columns=['Product', 'Date', 'Quantity'])}


# MySQL connection details
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='supermarket'
)

# Retrieve the product name from command line arguments
product_name = sys.argv[1]

# Define a dictionary to map product names to model file paths
model_mapping = {
    'Apple': './models/Apple_model.pkl',
    'Banana': './models/banana_model.pkl',
    'Bread': './models/bread_model.pkl',
    'Milk': './models/milk_model.pkl',
    'Orange': './models/orange_model.pkl',
    'Sweet': './models/sweet_model.pkl'
}

# Query to fetch data from the salesorder and product tables with the dynamically passed product name
query = f"""
    SELECT p.product_name, so.Date, p.Price, so.Quantity, so.CompetitorPrice, so.CompetitorCount, so.Promotion
    FROM salesorder AS so
    JOIN product AS p ON so.product_id = p.product_id
    WHERE p.product_name = '{product_name}'
"""

# Load the trained model based on the product name
model_path = model_mapping.get(product_name)
if model_path:
    with open(model_path, 'rb') as file:
        product_model = pickle.load(file)
else:
    print(f"Model not found for product: {product_name}")
    sys.exit(1)

# Create a cursor and execute the query
cursor = connection.cursor()
cursor.execute(query)

# Fetch the data and store it in a DataFrame
data = pd.DataFrame(cursor.fetchall(),
                    columns=['Product', 'Date', 'Price', 'Quantity', 'CompetitorPrice', 'CompetitorCount', 'Promotion'])

# Preprocess the data
preprocessed_data = preprocess_data(data, product_name)

# Make predictions using the loaded model and the preprocessed data
monthly_prediction_linear = product_model.predict(preprocessed_data['monthly'])
monthly_prediction_linear = [round(prediction) for prediction in monthly_prediction_linear]

# Get the maximum week and month values from the selected data
max_week = data['Date'].dt.isocalendar().week.max()
max_month = data['Date'].dt.month.max()

# Get the current year
current_year = data['Date'].dt.year.max()

# Check if the maximum week or month exceeds the maximum values of the current year
if max_week > pd.Timestamp(year=current_year, month=12, day=31).week:
    next_year = current_year + 1
else:
    next_year = current_year

# Generate the date ranges for the weekly and monthly predictions using the next year
date_range_monthly = pd.date_range(start=f'{next_year + 1}-01-01', periods=len(monthly_prediction_linear), freq='MS')

# Assign dates to the predicted values dataframes
predicted_df_monthly = pd.DataFrame({'Date': date_range_monthly, 'Prediction': monthly_prediction_linear})

# Convert the 'Date' column to a formatted string
predicted_df_monthly['Date'] = predicted_df_monthly['Date'].dt.strftime('%Y-%m-%d')

# Convert the 'Date' column to a datetime-like data type
predicted_df_monthly['Date'] = pd.to_datetime(predicted_df_monthly['Date'])

# Convert the 'Date' column to a formatted string with only the month name
predicted_df_monthly['Date'] = predicted_df_monthly['Date'].dt.strftime('%B')

# Convert DataFrames to JSON
json_data_monthly = predicted_df_monthly.to_json(orient='records')

# Create a dictionary to hold the JSON data
data = {
    'monthly': json.loads(json_data_monthly)
}

# Convert the dictionary to JSON
json_data = json.dumps(data)

# Print or return the JSON data
print(json_data)

# Close the cursor and connection
cursor.close()
connection.close()
