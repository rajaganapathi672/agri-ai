import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample training data (You should replace this with real crop yield dataset)
data = pd.DataFrame({
    'soil_type': [1, 2, 3, 1, 2, 3],
    'rainfall': [100, 120, 150, 80, 200, 180],
    'temperature': [25, 28, 30, 22, 35, 29],
    'humidity': [60, 70, 65, 55, 75, 68],
    'fertilizer_type': [1, 2, 3, 1, 2, 3],
    'crop_type': [1, 2, 3, 1, 2, 3],  # New column
    'yield': [2.5, 3.0, 3.8, 2.0, 3.5, 3.6]
})

# Features and target
X = data[['soil_type', 'rainfall', 'temperature', 'humidity', 'fertilizer_type', 'crop_type']]
y = data['yield']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, 'models/crop_yield_model.pkl')

print("✅ Model trained and saved as models/crop_yield_model.pkl")
