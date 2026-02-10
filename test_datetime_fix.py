"""Test prediction with the datetime fix"""
import sys
from datetime import datetime, timedelta
sys.path.append('.')

from src.app.model_service import ModelService

print("Testing model service with datetime fix...")
print("="*80)

# Initialize model service
model_service = ModelService()

# Create test input
test_input = {
    'airline': 'Biman Bangladesh Airlines',
    'source': 'DAC',
    'source_name': 'Dhaka',
    'destination': 'CXB',
    'destination_name': "Cox's Bazar",
    'departure_datetime': datetime.now() + timedelta(days=30),
    'arrival_datetime': datetime.now() + timedelta(days=30, hours=1),
    'stopovers': 'Direct',
    'aircraft_type': 'Boeing 737',
    'travel_class': 'Economy',
    'booking_source': 'Website',
}

print(f"\nTest Input:")
print(f"  Route: {test_input['source']} → {test_input['destination']}")
print(f"  Departure: {test_input['departure_datetime']}")
print(f"  Class: {test_input['travel_class']}")

# Make prediction
print(f"\nMaking prediction...")
try:
    predicted_fare, metadata = model_service.predict(test_input)
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS! Prediction completed without errors")
    print(f"{'='*80}")
    print(f"\nPredicted Fare: {predicted_fare:,.2f} BDT")
    print(f"Model Type: {metadata['model_type']}")
    print(f"Features Used: {metadata['features_used']}")
    print(f"\n{'='*80}")
    print(f"The datetime dtype error has been fixed!")
    print(f"{'='*80}")
except Exception as e:
    print(f"\n{'='*80}")
    print(f"❌ FAILED - Error occurred:")
    print(f"{'='*80}")
    print(f"{str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
