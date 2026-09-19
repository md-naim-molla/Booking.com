"""
Async Playwright web crawler for Booking.com with stealth and anti-bot evasion.
"""
import random
import asyncio
from urllib.parse import quote_plus
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
from scraper.parser import parse_property_card, parse_detail_page

class BookingScraper:
    def __init__(self, headless=config.HEADLESS):
        self.headless = headless
        self.stealth = Stealth()

    def build_search_url(self, city, checkin, checkout, offset=0, currency=config.CURRENCY):
        """Construct Booking.com search URL with parameters."""
        encoded_city = quote_plus(city)
        url = (
            f"https://www.booking.com/searchresults.html?"
            f"ss={encoded_city}&"
            f"checkin={checkin}&"
            f"checkout={checkout}&"
            f"group_adults={config.NUM_ADULTS}&"
            f"no_rooms={config.NUM_ROOMS}&"
            f"group_children={config.NUM_CHILDREN}&"
            f"selected_currency={currency}&"
            f"offset={offset}"
        )
        return url

    async def dismiss_popups(self, page):
        """Click away cookie banners, sign-in modals, and promotional overlays."""
        dismiss_selectors = [
            'button#onetrust-accept-btn-handler',
            'button[aria-label="Dismiss sign-in info."]',
            'button[aria-label="Close"]',
            'button.modal-mask-closeBtn',
            'button[data-testid="selection-modal-close-button"]'
        ]
        for sel in dismiss_selectors:
            try:
                btn = await page.query_selector(sel)
                if btn and await btn.is_visible():
                    await btn.click()
                    await asyncio.sleep(0.3)
            except Exception:
                pass

    async def safe_navigate(self, page, url, max_retries=3):
        """Navigate to URL with retry logic and graceful timeout handling."""
        for attempt in range(max_retries):
            try:
                await page.goto(url, wait_until='commit', timeout=25000)
                try:
                    await page.wait_for_load_state('domcontentloaded', timeout=15000)
                except Exception:
                    pass
                await asyncio.sleep(3.0)
                await self.dismiss_popups(page)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + random.uniform(1.0, 2.0))
                else:
                    print(f"\n[!] Navigation failed after {max_retries} attempts: {e}")
                    return False
        return False

    async def scrape_city_listings(self, city, checkin, checkout, max_hotels=config.MAX_HOTELS_PER_CITY):
        """
        Scrape search results for a single city across multiple pagination pages.
        Returns a list of parsed hotel dictionaries.
        """
        results = []
        offset = 0
        seen_ids = set()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled']
            )
            context = await browser.new_context(
                viewport={'width': config.VIEWPORT_WIDTH, 'height': config.VIEWPORT_HEIGHT},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                locale='en-US'
            )
            page = await context.new_page()
            await self.stealth.apply_stealth_async(page)

            # Warm up session with root domain
            try:
                await page.goto("https://www.booking.com", wait_until='commit', timeout=15000)
                await asyncio.sleep(1.5)
                await self.dismiss_popups(page)
            except Exception:
                pass

            pbar = tqdm(total=max_hotels if max_hotels else 500, desc=f"Scraping {city}")

            while True:
                if max_hotels and len(results) >= max_hotels:
                    break

                search_url = self.build_search_url(city, checkin, checkout, offset=offset)
                success = await self.safe_navigate(page, search_url)
                
                if not success:
                    break

                # Scroll down to trigger lazy loading
                try:
                    await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.7);")
                    await asyncio.sleep(1.5)
                except Exception:
                    pass

                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                cards = soup.find_all('div', {'data-testid': 'property-card'})

                if not cards:
                    cards = soup.select('.sr_property_block')

                if not cards:
                    print(f"\n[!] No property cards found for {city} at offset {offset}.")
                    break

                new_hotels_in_page = 0
                for card in cards:
                    hotel_data = parse_property_card(card, city, checkin, checkout)
                    h_id = hotel_data.get('hotel_id') or hotel_data.get('hotel_name')
                    
                    if h_id and h_id not in seen_ids and hotel_data.get('price_net'):
                        seen_ids.add(h_id)
                        results.append(hotel_data)
                        new_hotels_in_page += 1
                        pbar.update(1)
                        
                        if max_hotels and len(results) >= max_hotels:
                            break

                if new_hotels_in_page == 0:
                    print(f"\n[i] Reached end of distinct listings for {city}.")
                    break

                offset += config.PAGE_SIZE
                delay = random.uniform(config.MIN_DELAY, config.MAX_DELAY)
                await asyncio.sleep(delay)

            pbar.close()
            await browser.close()

        return results

    async def enrich_with_details(self, hotel_records, max_concurrent=4):
        """
        Enrich listing records by visiting hotel detail pages to extract
        exact GPS, sub-ratings, and detailed facility flags.
        """
        enriched = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_detail(hotel, context, pbar):
            url = hotel.get('hotel_url')
            if not url:
                enriched.append(hotel)
                pbar.update(1)
                return

            async with semaphore:
                page = await context.new_page()
                try:
                    await self.stealth.apply_stealth_async(page)
                    await page.goto(url, wait_until='commit', timeout=20000)
                    try:
                        await page.wait_for_load_state('domcontentloaded', timeout=10000)
                    except Exception:
                        pass
                    await asyncio.sleep(1.5)
                    await self.dismiss_popups(page)
                    
                    html = await page.content()
                    detailed_data = parse_detail_page(html, hotel)
                    enriched.append(detailed_data)
                except Exception:
                    enriched.append(hotel)
                finally:
                    await page.close()
                    pbar.update(1)
                    await asyncio.sleep(random.uniform(0.5, 1.5))

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled']
            )
            context = await browser.new_context(
                viewport={'width': config.VIEWPORT_WIDTH, 'height': config.VIEWPORT_HEIGHT},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                locale='en-US'
            )
            
            pbar = tqdm(total=len(hotel_records), desc="Extracting Deep Details & Sub-scores")
            tasks = [fetch_detail(h, context, pbar) for h in hotel_records]
            await asyncio.gather(*tasks)
            pbar.close()
            await browser.close()

        return enriched

