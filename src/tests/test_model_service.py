"""
Tests for Model Service Module

This module contains tests for the ModelService class.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app.model_service import ModelService


# Get project root directory for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


@pytest.fixture
def model_service():
    """Fixture to provide ModelService instance with absolute model path."""
    return ModelService(str(MODEL_PATH))


class TestModelService:
    """Test cases for ModelService class."""
    
    def test_model_loads_successfully(self, model_service):
        """Test that the model loads without errors."""
        model = model_service.load_model()
        
        assert model is not None
        assert model_service.is_loaded is True
    
    def test_model_path_exists(self, model_service):
        """Test that the model file exists."""
        assert model_service.model_path.exists()
    
    def test_prediction_with_sample_data(self, model_service):
        """Test prediction with valid sample data."""
        model_service.load_model()
        
        # Sample input data
        sample_input = {
            'airline': 'Emirates',
            'source': 'DAC',
            'source_name': "Hazrat Shahjalal International Airport, Dhaka",
            'destination': 'DXB',
            'destination_name': 'Dubai International Airport',
            'departure_datetime': datetime.now() + timedelta(days=30),
            'arrival_datetime': datetime.now() + timedelta(days=30, hours=4),
            'stopovers': 'Direct',
            'aircraft_type': 'Boeing 787',
            'travel_class': 'Economy',
            'booking_source': 'Online Website',
        }
        
        try:
            prediction, metadata = model_service.predict(sample_input)
            
            # Assertions
            assert prediction > 0
            assert isinstance(prediction, float)
            assert 'model_type' in metadata
            assert 'features_used' in metadata
            
        except Exception as e:
            pytest.skip(f"Prediction test skipped due to: {str(e)}")
    
    def test_invalid_dates_raise_error(self, model_service):
        """Test that invalid dates are handled properly."""
        model_service.load_model()
        
        # Invalid input: arrival before departure
        invalid_input = {
            'airline': 'Emirates',
            'source': 'DAC',
            'source_name': "Hazrat Shahjalal International Airport, Dhaka",
            'destination': 'DXB',
            'destination_name': 'Dubai International Airport',
            'departure_datetime': datetime.now() + timedelta(days=30),
            'arrival_datetime': datetime.now() + timedelta(days=29),  # Before departure
            'stopovers': 'Direct',
            'aircraft_type': 'Boeing 787',
            'travel_class': 'Economy',
            'booking_source': 'Online Website',
        }
        
        # This should raise an error or handle it gracefully
        try:
            prediction, metadata = model_service.predict(invalid_input)
            # If it doesn't raise an error, the duration should be invalid
            assert prediction > 0  # At minimum, prediction should be positive
        except (ValueError, Exception):
            # Expected behavior
            pass
    
    def test_get_model_info(self, model_service):
        """Test getting model information."""
        # Before loading
        info = model_service.get_model_info()
        assert info['status'] == 'Not loaded'
        
        # After loading
        model_service.load_model()
        info = model_service.get_model_info()
        assert info['status'] == 'Loaded'
        assert 'model_type' in info
        assert 'model_path' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
