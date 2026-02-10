"""Quick script to inspect model's expected features"""
import sys
sys.path.append('.')

import pickle
import pandas as pd

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model type:", type(model).__name__)
print("\n" + "="*80)
print("EXPECTED FEATURES:")
print("="*80)

if hasattr(model, 'feature_names_in_'):
    features = model.feature_names_in_
    print(f"\nTotal features: {len(features)}\n")
    for i, feat in enumerate(features, 1):
        print(f"{i:2d}. {feat}")
else:
    print("Model doesn't store feature names")

print("\n" + "="*80)
print("Sample from training data:")
print("="*80)
df = pd.read_csv('data/raw/Flight_Price_Dataset_of_Bangladesh.csv', nrows=1)
print("\nColumns in CSV:")
for col in df.columns:
    print(f"  - {col}")
