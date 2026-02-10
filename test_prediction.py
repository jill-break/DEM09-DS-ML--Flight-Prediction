"""
Quick test to verify model service works with updated features
"""
import sys
from datetime import datetime, timedelta
sys.path.append('.')

from src.app.model_service import ModelService

# Initialize model service
print("Initializing model service...")
model_service = ModelService()

# Load model
print("Loading model...")
model = model_service.load_model()
print(f"✓ Model loaded successfully: {type(model).__name__}")

# Create test input
print("\nCreating test input...")
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

# Make prediction
print("\nMaking prediction...")
try:
    predicted_fare, metadata = model_service.predict(test_input)
    print(f"✓ Prediction successful!")
    print(f"\nPredicted Fare: {predicted_fare:,.2f} BDT")
    print(f"Model Type: {metadata['model_type']}")
    print(f"Features Used: {metadata['features_used']}")
    print(f"Confidence: {metadata['confidence']}")
    print("\n✅ ALL TESTS PASSED - Model service is working correctly!")
except Exception as e:
    print(f"✗ Prediction failed with error:")
    print(f"  {str(e)}")
    print("\n❌ TEST FAILED")
    sys.exit(1)
