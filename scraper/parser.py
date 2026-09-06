"""
HTML and JSON-LD parser for Booking.com property cards and detail pages.
"""
import re
import json
import math
from datetime import datetime
from bs4 import BeautifulSoup

def clean_number(text):
    """Extract float or int from messy text (e.g. 'US$ 1,245.50' -> 1245.5)."""
    if not text:
        return None
    cleaned = re.sub(r'[^\d.]', '', text.replace(',', ''))
    try:
        val = float(cleaned)
        return val
    except (ValueError, TypeError):
        return None

def extract_distance_km(text):
    """Convert distance strings like '7.3 km from downtown' or '500 m from center' to float in km."""
    if not text:
        return None
    text_lower = text.lower()
    
    # Check for meters
    m_match = re.search(r'([\d.]+)\s*m\b', text_lower)
    if m_match and 'km' not in text_lower:
        try:
            return round(float(m_match.group(1)) / 1000.0, 3)
        except ValueError:
            pass
            
    # Check for km
    km_match = re.search(r'([\d.]+)\s*km\b', text_lower)
    if km_match:
        try:
            return float(km_match.group(1))
        except ValueError:
            pass
            
    num = clean_number(text)
    return num if num is not None else None

def parse_property_card(card, city, checkin, checkout):
    """
    Parse a single property card from search result pages (SERP).
    Returns a structured dictionary of ML features.
    """
    data = {}
    
    # Basic Query Metadata
    data['city'] = city
    data['checkin_date'] = checkin
    data['checkout_date'] = checkout
    data['scrape_timestamp'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate Lead Time and Weekend flag
    try:
        cin = datetime.strptime(checkin, "%Y-%m-%d")
        now = datetime.utcnow()
        data['lead_time_days'] = max(0, (cin - now).days)
        data['is_weekend'] = 1 if cin.weekday() in [4, 5] else 0  # Fri/Sat check-in
    except Exception:
        data['lead_time_days'] = 0
        data['is_weekend'] = 0

    # 1. Title & URL
    title_el = card.find(attrs={'data-testid': 'title'})
    data['hotel_name'] = title_el.text.strip() if title_el else None
    
    link_el = card.find(attrs={'data-testid': 'title-link'}) or card.find('a', href=re.compile(r'/hotel/'))
    if link_el and link_el.get('href'):
        raw_href = link_el['href']
        data['hotel_url'] = raw_href.split('?')[0] if '?' in raw_href else raw_href
        id_match = re.search(r'hotelid=(\d+)', raw_href) or re.search(r'/hotel/[^/]+/([^.]+)', raw_href)
        data['hotel_id'] = id_match.group(1) if id_match else data['hotel_name']
    else:
        data['hotel_url'] = None
        data['hotel_id'] = data['hotel_name']

    # 2. Target Variable: Price (Y)
    price_el = card.find(attrs={'data-testid': 'price-and-discounted-price'})
    data['price_net'] = clean_number(price_el.text) if price_el else None
    data['log_price'] = round(math.log(data['price_net']), 4) if (data['price_net'] and data['price_net'] > 0) else None

    # Reference / Strikethrough Price
    orig_price_el = card.find(attrs={'data-testid': 'original-price'}) or card.find('span', class_=re.compile(r'strikethrough|strike-through'))
    data['price_original'] = clean_number(orig_price_el.text) if orig_price_el else data['price_net']
    
    if data['price_original'] and data['price_net'] and data['price_original'] > data['price_net']:
        data['discount_pct'] = round((data['price_original'] - data['price_net']) / data['price_original'], 4)
    else:
        data['discount_pct'] = 0.0

    # Mandatory Taxes and Charges
    taxes_el = card.find(attrs={'data-testid': 'taxes-and-charges'})
    data['taxes_and_charges'] = clean_number(taxes_el.text) if taxes_el else 0.0

    # 3. Quality & Star Rating
    star_match = None
    aria_el = card.find(attrs={'aria-label': re.compile(r'(\d+)\s*out of\s*5\s*(stars|squares)', re.I)})
    if aria_el:
        m = re.search(r'(\d+)\s*out of', aria_el.get('aria-label', ''))
        if m:
            star_match = float(m.group(1))
            
    if not star_match:
        rating_container = card.find(attrs={'data-testid': re.compile(r'rating-(stars|squares)')})
        if rating_container:
            # Check aria-label on rating container
            c_aria = rating_container.get('aria-label', '')
            m = re.search(r'(\d+)\s*out of', c_aria)
            if m:
                star_match = float(m.group(1))
            else:
                # Count star spans/svgs
                spans = rating_container.find_all('span', recursive=False)
                star_match = float(len(spans)) if spans else 0.0

    data['star_rating'] = star_match if star_match is not None else 0.0

    # 4. Review Score & Popularity Signals
    score_el = card.find(attrs={'data-testid': 'review-score'})
    if score_el:
        score_text = score_el.text
        score_match = re.search(r'\b([1-9]\.\d|10(?:\.0)?)\b', score_text)
        data['review_score_overall'] = float(score_match.group(1)) if score_match else None
        
        rev_count_match = re.search(r'([\d,]+)\s+reviews', score_text, re.IGNORECASE)
        data['review_count'] = int(rev_count_match.group(1).replace(',', '')) if rev_count_match else 0
        
        for label in ['Superb', 'Exceptional', 'Wonderful', 'Fabulous', 'Very Good', 'Good', 'Pleasant']:
            if label.lower() in score_text.lower():
                data['review_score_label'] = label
                break
        if 'review_score_label' not in data:
            data['review_score_label'] = 'Standard'
    else:
        data['review_score_overall'] = None
        data['review_count'] = 0
        data['review_score_label'] = 'No reviews'

    # 5. Spatial & Location Attributes
    dist_el = card.find(attrs={'data-testid': 'distance'})
    data['distance_to_center_km'] = extract_distance_km(dist_el.text) if dist_el else None
    
    loc_el = card.find(attrs={'data-testid': 'address'}) or card.find(attrs={'data-testid': 'address-link'})
    data['neighborhood'] = loc_el.text.strip() if loc_el else city

    # 6. Amenity & Policy Signals
    card_text = card.text.lower()
    
    data['breakfast_included'] = 1 if 'breakfast included' in card_text or 'free breakfast' in card_text else 0
    data['free_cancellation'] = 1 if 'free cancellation' in card_text else 0
    data['is_beachfront'] = 1 if 'beachfront' in card_text or 'private beach' in card_text else 0
    data['has_free_wifi'] = 1 if 'free wifi' in card_text or 'free internet' in card_text else 0
    data['has_swimming_pool'] = 1 if 'pool' in card_text or 'swimming pool' in card_text else 0
    data['has_parking'] = 1 if 'free parking' in card_text or 'parking available' in card_text or 'parking' in card_text else 0
    data['has_air_conditioning'] = 1 if 'air conditioning' in card_text or 'ac' in card_text else 0
    
    # Urgency & Scarcity Signals
    scarcity_match = re.search(r'only\s*(\d+)\s*left', card_text)
    data['rooms_left_warning'] = int(scarcity_match.group(1)) if scarcity_match else 0
    
    # Sustainability
    sust_el = card.find(attrs={'data-testid': 'sustainability-level'})
    data['sustainability_level'] = sust_el.text.strip() if sust_el else 'None'
    
    # Property Type
    if 'apartment' in card_text:
        data['property_type'] = 'Apartment'
    elif 'resort' in card_text:
        data['property_type'] = 'Resort'
    elif 'villa' in card_text:
        data['property_type'] = 'Villa'
    elif 'hostel' in card_text:
        data['property_type'] = 'Hostel'
    elif 'guest house' in card_text or 'guesthouse' in card_text:
        data['property_type'] = 'Guesthouse'
    else:
        data['property_type'] = 'Hotel'

    return data

def parse_detail_page(page_html, base_data):
    """
    Extract granular sub-ratings, exact GPS, and detailed facility matrix
    from a hotel's detail page. Merges with base_data.
    """
    soup = BeautifulSoup(page_html, 'lxml')
    merged = dict(base_data)
    
    # 1. Parse JSON-LD structured data for exact GPS Coordinates
    json_scripts = soup.find_all('script', type='application/ld+json')
    for s in json_scripts:
        try:
            if not s.string:
                continue
            d = json.loads(s.string)
            if isinstance(d, dict) and d.get('@type') in ['Hotel', 'LodgingBusiness', 'Resort', 'Hostel', 'Apartment', 'BedAndBreakfast']:
                geo = d.get('geo', {})
                if geo:
                    merged['latitude'] = float(geo.get('latitude')) if geo.get('latitude') else None
                    merged['longitude'] = float(geo.get('longitude')) if geo.get('longitude') else None
                addr = d.get('address', {})
                if isinstance(addr, dict):
                    merged['full_address'] = addr.get('streetAddress', '')
                if d.get('starRating') and isinstance(d.get('starRating'), dict):
                    merged['star_rating'] = float(d['starRating'].get('ratingValue', merged.get('star_rating', 0)))
                break
        except Exception:
            pass

    # Fallback coordinates from script tags if JSON-LD missing
    if 'latitude' not in merged or merged.get('latitude') is None:
        lat_match = re.search(r'b_map_center_latitude\s*[:=]\s*([-\d.]+)', page_html)
        lng_match = re.search(r'b_map_center_longitude\s*[:=]\s*([-\d.]+)', page_html)
        if lat_match and lng_match:
            try:
                merged['latitude'] = float(lat_match.group(1))
                merged['longitude'] = float(lng_match.group(1))
            except ValueError:
                pass

    # 2. Extract 7 Granular Sub-Scores (Cleanliness, Staff, Facilities, Comfort, Value, Location, WiFi)
    sub_scores = {
        'score_cleanliness': None,
        'score_staff': None,
        'score_facilities': None,
        'score_comfort': None,
        'score_value': None,
        'score_location': None,
        'score_wifi': None
    }
    
    review_sub_blocks = soup.find_all(attrs={'data-testid': re.compile(r'review-subscore|subscore')}) or soup.find_all('div', class_=re.compile(r'review_score_breakdown|subscore|review-subscore|a53cb322e1'))
    for block in review_sub_blocks:
        b_text = block.text.lower()
        num = clean_number(b_text)
        if num and 1.0 <= num <= 10.0:
            if 'cleanliness' in b_text and not sub_scores['score_cleanliness']:
                sub_scores['score_cleanliness'] = num
            elif 'staff' in b_text and not sub_scores['score_staff']:
                sub_scores['score_staff'] = num
            elif 'facilities' in b_text and not sub_scores['score_facilities']:
                sub_scores['score_facilities'] = num
            elif 'comfort' in b_text and not sub_scores['score_comfort']:
                sub_scores['score_comfort'] = num
            elif ('value' in b_text or 'money' in b_text) and not sub_scores['score_value']:
                sub_scores['score_value'] = num
            elif 'location' in b_text and not sub_scores['score_location']:
                sub_scores['score_location'] = num
            elif ('wifi' in b_text or 'wi-fi' in b_text or 'internet' in b_text) and not sub_scores['score_wifi']:
                sub_scores['score_wifi'] = num

    merged.update(sub_scores)

    # 3. Granular Amenities Matrix (Binary Flags)
    full_text = page_html.lower()
    merged['has_spa'] = 1 if 'spa' in full_text or 'wellness centre' in full_text or 'sauna' in full_text else 0
    merged['has_fitness_center'] = 1 if 'fitness center' in full_text or 'gym' in full_text or 'fitness' in full_text else 0
    merged['has_restaurant'] = 1 if 'restaurant' in full_text or 'dining' in full_text else 0
    merged['has_bar'] = 1 if 'bar' in full_text or 'lounge' in full_text else 0
    merged['has_airport_shuttle'] = 1 if 'airport shuttle' in full_text or 'airport transfer' in full_text else 0
    merged['has_24h_front_desk'] = 1 if '24-hour front desk' in full_text or '24-hour reception' in full_text else 0
    merged['pets_allowed'] = 1 if 'pets are allowed' in full_text or 'pet friendly' in full_text else 0
    
    # Photo gallery count
    img_tags = soup.find_all('img', class_=re.compile(r'hotel_image|gallery|bh-photo'))
    merged['photo_count'] = max(len(img_tags), 1)

    return merged
