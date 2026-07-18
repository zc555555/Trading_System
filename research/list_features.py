"""List all features from the trained model."""
import pickle
from pathlib import Path

model_path = Path(__file__).parent / "artifacts" / "ensemble_time_windows.pkl"

with open(model_path, 'rb') as f:
    pkl = pickle.load(f)

features = pkl['feature_cols']

print(f"Total features: {len(features)}")
print("\n" + "="*80)
print("ALL FEATURES:")
print("="*80)

for i, feat in enumerate(features, 1):
    print(f"{i:3}. {feat}")
