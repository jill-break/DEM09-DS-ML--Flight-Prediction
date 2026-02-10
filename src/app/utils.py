"""
Utility functions for the web application.

This module provides helper functions for data validation, formatting,
and constants used throughout the web app.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

# Flight data constants
AIRLINES = [
    'Malaysian Airlines', 'Cathay Pacific', 'British Airways', 'Singapore Airlines',
    'IndiGo', 'US-Bangla Airlines', 'AirAsia', 'Etihad Airways', 'Gulf Air',
    'Emirates', 'Air India', 'Saudia', 'Thai Airways', 'Kuwait Airways',
    'FlyDubai', 'NovoAir', 'Air Astra', 'SriLankan Airlines', 'Biman Bangladesh Airlines',
    'Qatar Airways', 'Vistara', 'Turkish Airlines', 'Lufthansa', 'Air Arabia'
]

AIRPORTS = {
    'CXB': "Cox's Bazar Airport",
    'CCU': "Netaji Subhas Chandra Bose International Airport, Kolkata",
    'BZL': 'Barisal Airport',
    'CGP': "Shah Amanat International Airport, Chittagong",
    'ZYL': "Osmani International Airport, Sylhet",
    'KUL': 'Kuala Lumpur International Airport',
    'SPD': 'Saidpur Airport',
    'YYZ': 'Toronto Pearson International Airport',
    'RJH': "Shah Makhdum Airport, Rajshahi",
    'DAC': "Hazrat Shahjalal International Airport, Dhaka",
    'JSR': 'Jessore Airport',
    'LHR': 'London Heathrow Airport',
    'DEL': "Indira Gandhi International Airport, Delhi",
    'IST': 'Istanbul Airport',
    'DXB': 'Dubai International Airport',
    'SIN': 'Singapore Changi Airport',
    'BKK': "Suvarnabhumi Airport, Bangkok",
    'DOH': "Hamad International Airport, Doha",
    'JFK': "John F. Kennedy International Airport, New York",
    'JED': "King Abdulaziz International Airport, Jeddah"
}

AIRCRAFT_TYPES = ['Airbus A320', 'Boeing 787', 'Boeing 737', 'Airbus A350', 'Boeing 777']

CLASSES = ['Economy', 'Business', 'First Class']

BOOKING_SOURCES = ['Online Website', 'Travel Agency', 'Direct Booking']

STOPOVERS = ['Direct', '1 Stop', '2 Stops']

SEASONALITY = ['Regular', 'Winter Holidays', 'Eid', 'Hajj']


def validate_date(date_value: datetime) -> bool:
    """
    Validate that the date is in the future.
    
    Args:
        date_value: Date to validate
        
    Returns:
        True if valid, False otherwise
    """
    return date_value >= datetime.now()


def calculate_duration(departure: datetime, arrival: datetime) -> float:
    """
    Calculate flight duration in hours.
    
    Args:
        departure: Departure datetime
        arrival: Arrival datetime
        
    Returns:
        Duration in hours
    """
    duration = arrival - departure
    return duration.total_seconds() / 3600


def calculate_days_before_departure(departure: datetime) -> int:
    """
    Calculate days before departure from today.
    
    Args:
        departure: Departure datetime
        
    Returns:
        Number of days until departure
    """
    delta = departure - datetime.now()
    return max(0, delta.days)


def determine_seasonality(departure_date: datetime) -> str:
    """
    Determine seasonality based on departure date.
    
    Args:
        departure_date: Departure datetime
        
    Returns:
        Seasonality category
    """
    month = departure_date.month
    
    # Winter holidays: December and January
    if month in [12, 1]:
        return 'Winter Holidays'
    # Eid: Approximate (March-April)
    elif month in [3, 4]:
        return 'Eid'
    # Hajj: June timeframe
    elif month == 6:
        return 'Hajj'
    else:
        return 'Regular'


def format_currency(amount: float) -> str:
    """
    Format amount as BDT currency.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    return f"৳{amount:,.2f}"


def create_input_dataframe(user_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert user input dictionary to DataFrame format expected by model.
    
    Args:
        user_input: Dictionary of user inputs
        
    Returns:
        DataFrame with single row
    """
    return pd.DataFrame([user_input])


def get_airport_name(code: str) -> str:
    """
    Get full airport name from code.
    
    Args:
        code: Airport code
        
    Returns:
        Full airport name
    """
    return AIRPORTS.get(code, code)
