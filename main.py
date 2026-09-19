"""
Booking.com Scraper Orchestrator for Large-Scale Machine Learning Publication Dataset (~4,000 observations).
"""
import os
import argparse
import asyncio
import pandas as pd
from datetime import datetime

import config
from scraper.crawler import BookingScraper

def parse_args():
    parser = argparse.ArgumentParser(description="Booking.com Research Scraper for Large-Scale ML Dataset Generation")
    parser.add_argument("--destinations", nargs="+", default=config.DESTINATIONS, help="List of cities to scrape")
    parser.add_argument("--limit", type=int, default=config.MAX_HOTELS_PER_CITY, help="Max hotels per city per window")
    parser.add_argument("--currency", type=str, default=config.CURRENCY, help="Currency (USD, EUR, BDT)")
    parser.add_argument("--skip-details", action="store_true", help="Skip detail page visits (faster)")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    return parser.parse_args()

async def main():
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    destinations = ["Dubai"] if args.test else args.destinations
    limit = 10 if args.test else args.limit
    date_windows = [config.DATE_WINDOWS[0]] if args.test else config.DATE_WINDOWS
    
    print("=" * 75)
    print("🎓 BOOKING.COM LARGE-SCALE DATASET GENERATOR (~4,000 ROWS FOR Q1 PUBLICATION)")
    print("=" * 75)
    print(f"• Total Destinations: {len(destinations)} ({', '.join(destinations[:6])}...)")
    print(f"• Lead-Time Windows:  {len(date_windows)} ({', '.join(w['name'] for w in date_windows)})")
    print(f"• Target Volume:      ~{len(destinations) * len(date_windows) * limit} total observations")
    print(f"• Target Currency:    {args.currency}")
    print(f"• Output Path:        {config.FINAL_DATASET_CSV}")
    print("=" * 75)

    scraper = BookingScraper(headless=config.HEADLESS)
    
    # Load existing records if any
    all_hotels = []
    if os.path.exists(config.FINAL_DATASET_CSV):
        try:
            existing_df = pd.read_csv(config.FINAL_DATASET_CSV)
            all_hotels = existing_df.to_dict('records')
            print(f"[i] Loaded {len(all_hotels)} existing observations from previous runs.")
        except Exception:
            all_hotels = []

    for window in date_windows:
        w_name = window["name"]
        checkin = window["checkin"]
        checkout = window["checkout"]
        print(f"\n===========================================================================")
        print(f"📅 PROCESSING STAY WINDOW: {w_name} (Check-in: {checkin}, Check-out: {checkout})")
        print(f"===========================================================================")

        for city in destinations:
            print(f"\n[+] Searching listings for {city} [{w_name}]...")
            city_records = await scraper.scrape_city_listings(
                city=city,
                checkin=checkin,
                checkout=checkout,
                max_hotels=limit
            )
            print(f"  ✓ Collected {len(city_records)} listings for {city} ({w_name}).")

            if not args.skip_details and city_records:
                print(f"[+] Extracting Deep Details (GPS, Sub-scores, Amenities) for {city}...")
                city_records = await scraper.enrich_with_details(city_records)

            all_hotels.extend(city_records)

            # Continuous Incremental Save
            if all_hotels:
                temp_df = pd.DataFrame(all_hotels)
                temp_df = temp_df.drop_duplicates(subset=['hotel_name', 'city', 'checkin_date'])
                temp_df.to_csv(config.FINAL_DATASET_CSV, index=False)
                print(f"  💾 Continuous checkpoint: {len(temp_df)} total observations saved to {config.FINAL_DATASET_CSV}")

    # Final Data Processing & Clean Type Casting
    df = pd.DataFrame(all_hotels)
    df = df.drop_duplicates(subset=['hotel_name', 'city', 'checkin_date'])
    
    numeric_cols = [
        'price_net', 'log_price', 'price_original', 'discount_pct', 'taxes_and_charges',
        'star_rating', 'review_score_overall', 'review_count', 'distance_to_center_km',
        'breakfast_included', 'free_cancellation', 'is_beachfront', 'has_free_wifi',
        'has_swimming_pool', 'has_parking', 'has_air_conditioning', 'rooms_left_warning',
        'lead_time_days', 'is_weekend', 'latitude', 'longitude', 'score_cleanliness',
        'score_staff', 'score_facilities', 'score_comfort', 'score_value', 'score_location',
        'score_wifi', 'has_spa', 'has_fitness_center', 'has_restaurant', 'has_bar',
        'has_airport_shuttle', 'has_24h_front_desk', 'pets_allowed', 'photo_count'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv(config.FINAL_DATASET_CSV, index=False)
    
    print("\n" + "=" * 75)
    print("✅ LARGE-SCALE DATASET GENERATION COMPLETE!")
    print("=" * 75)
    print(f"• Total Rows (N):      {len(df)}")
    print(f"• Total Features (P):  {len(df.columns)}")
    print(f"• Target Variable (Y): price_net (Min: ${df['price_net'].min():.2f}, Max: ${df['price_net'].max():.2f}, Mean: ${df['price_net'].mean():.2f})")
    print(f"• File Location:       {config.FINAL_DATASET_CSV}")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(main())
