# house-price-prediction
# House-Price-Prediction
PYTHON CODE import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("kc_house_data.csv")

# Basic features selection (customize based on your data)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built']

target = 'price'

# Handle missing data if any
df = df.dropna(subset=features + [target])

X = df[features] y = df[target]

# Normalize the features
scaler = StandardScaler() X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) from tensorflow.keras import Input from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense

model = Sequential([ Input(shape=(X_train.shape[1],)), Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(32, activation='relu'), Dense(1) ])

model.compile(optimizer='adam', loss='mse', metrics=['mae']) model.summary() history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1) import matplotlib.pyplot as plt

# Evaluate model
loss, mae = model.evaluate(X_test, y_test) print(f"Test MAE: ${mae:,.2f}")

# Plot loss
plt.plot(history.history['mae'], label='Train MAE') plt.plot(history.history['val_mae'], label='Val MAE') plt.title("Training vs Validation MAE") plt.xlabel("Epochs") plt.ylabel("MAE") plt.legend() plt.grid(True) plt.tight_layout() plt.show()

# Features list
import pandas as pd

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built']

# Function to get input
def get_user_input(): print("\n📥 Enter the house features below:") user_data = [] for feature in features: val = input(f"{feature}: ") try: val = float(val) except ValueError: print("Invalid input. Please enter a number.") return None user_data.append(val) return [user_data]

# Prediction function
def predict_price(sample_input): sample_df = pd.DataFrame(sample_input, columns=features) sample_scaled = scaler.transform(sample_df) prediction = model.predict(sample_scaled)[0][0] return prediction

# --- Run prediction ---
user_input = get_user_input() if user_input: predicted = predict_price(user_input) print(f"\n🏠 Predicted House Price: ${predicted:,.2f}")

CSV FILE https://www.kaggle.com/datasets/shivachandel/kc-house-data
