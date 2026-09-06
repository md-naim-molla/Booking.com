"""
Configuration settings for Booking.com ML Research Scraper.
"""
from datetime import datetime, timedelta

# Expanded destination list across global tourism typologies
DESTINATIONS = [
    # Metropole & Business Hubs
    "London", "Paris", "Tokyo", "Singapore", "New York", "Berlin", "Amsterdam",
    # Luxury & Resort Hubs
    "Dubai", "Rome", "Barcelona", "Madrid", "Vienna", "Milan", "Abu Dhabi",
    # Southeast Asian & Leisure Hubs
    "Bangkok", "Phuket", "Bali", "Kuala Lumpur",
    # South Asian & Emerging Tourism Hubs
    "Cox's Bazar", "Dhaka", "Chittagong", "Sylhet"
]

# Multiple Lead-Time Windows (Panel Dataset for Dynamic Pricing & Seasonality)
now = datetime.utcnow()
DATE_WINDOWS = [
    {
        "name": "30-Day Lead Time",
        "checkin": (now + timedelta(days=30)).strftime("%Y-%m-%d"),
        "checkout": (now + timedelta(days=31)).strftime("%Y-%m-%d")
    },
    {
        "name": "60-Day Lead Time",
        "checkin": (now + timedelta(days=60)).strftime("%Y-%m-%d"),
        "checkout": (now + timedelta(days=61)).strftime("%Y-%m-%d")
    },
    {
        "name": "90-Day Lead Time",
        "checkin": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
        "checkout": (now + timedelta(days=91)).strftime("%Y-%m-%d")
    }
]

DEFAULT_CHECKIN = DATE_WINDOWS[0]["checkin"]
DEFAULT_CHECKOUT = DATE_WINDOWS[0]["checkout"]

# Guest Context
NUM_ADULTS = 2
NUM_ROOMS = 1
NUM_CHILDREN = 0
CURRENCY = "USD"

# Scraping Limits per Destination per Date Window
MAX_HOTELS_PER_CITY = 100
PAGE_SIZE = 25

# Rate Limiting & Delays (seconds)
MIN_DELAY = 1.5
MAX_DELAY = 3.0
PAGE_LOAD_TIMEOUT = 30000  # ms
PAGE_WAIT_TIME = 2.5       # seconds for dynamic rendering

# Browser Configuration
HEADLESS = True
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080

# Output Paths
OUTPUT_DIR = "data"
FINAL_DATASET_CSV = "data/hotel_prediction_dataset.csv"
RAW_CACHE_CSV = "data/hotels_raw_cache.csv"

