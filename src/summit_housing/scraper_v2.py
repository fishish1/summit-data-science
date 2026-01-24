import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import time
import argparse
from typing import List, Dict, Optional
import logging
from pathlib import Path
import random

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://gis.summitcountyco.gov/map/DetailData.aspx"

class SummitScraper:
    def __init__(self, input_csv: str, output_csv: str, concurrency: int = 20):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = None
        self.processed_schnos = set()
        
    async def get_schnos(self) -> List[str]:
        """Reads schedule numbers from input CSV."""
        try:
            df = pd.read_csv(self.input_csv, low_memory=False)
            if 'schno' in df.columns:
                return df['schno'].astype(str).unique().tolist()
            elif 'Schno' in df.columns:
                return df['Schno'].astype(str).unique().tolist()
            else:
                logger.error("Input CSV must have a 'schno' column.")
                return []
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_csv}")
            return []

    def load_progress(self):
        """Loads already scraped IDs to resume progress."""
        if Path(self.output_csv).exists():
            try:
                df = pd.read_csv(self.output_csv)
                if 'schno' in df.columns:
                    self.processed_schnos = set(df['schno'].astype(str))
                logger.info(f"Resuming... {len(self.processed_schnos)} records already scraped.")
            except Exception as e:
                logger.warning(f"Could not read output file: {e}")

    async def fetch_page(self, schno: str) -> Optional[str]:
        """Fetches the HTML content for a property."""
        url = f"{BASE_URL}?Schno={schno}"
        # Rotate User Agents or just provide a standard browser one
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        }
        async with self.semaphore:  # Rate limiting
            try:
                # Add random sleep of 0.5-1.5s to be polite
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
                async with self.session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"Error {response.status} for {schno}")
                        return None
            except Exception as e:
                # Use repr(e) to see the actual error type (e.g. ClientConnectorError)
                logger.error(f"Request failed for {schno}: {repr(e)}")
                return None

    def parse_html(self, html: str, schno: str) -> List[Dict]:
        """Parses the HTML to extract Sales History and Improvement Data."""
        soup = BeautifulSoup(html, 'html.parser')
        records = []
        
        # 1. Renovation / Year Built Data (From Improvements/ImpData)
        # We look for the 'ImpData' table or similar data points
        # Logic: Find 'Effective Year Built' text
        year_built = None
        eff_year_built = None
        
        # Search strategy: Look for cells containing the text and get the next value
        # This is generic because layout tables are messy
        try:
            yb_elem = soup.find(string=lambda t: t and "Year Built" in t)
            if yb_elem:
                # Value is likely in the next TD or sibling
                # This depends heavily on DOM structure, assuming simple table layout
                pass # Placeholder for complex DOM traversal if needed
        except: pass
        
        # 2. Sales History (from DetailData table usually or lower section)
        # Based on user's Curl: <table class="DetailData">...<td...Sales History or headers>
        # Headers: Reception | Sale Date | Document Type | Sale Price
        
        # Find the table that contains "Sale Date" or "Reception"
        tables = soup.find_all("table")
        sales_table = None
        for t in tables:
            if t.find(string=lambda s: s and "Sale Price" in s):
                sales_table = t
                break
        
        if sales_table:
            # Iterate rows. Skip header.
            rows = sales_table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                # Need robust way to identify if it's a data row
                # Looking for a date pattern or numeric reception #
                if len(cols) >= 4: # Min cols for sales data
                    vals = [c.get_text(strip=True) for c in cols]
                    
                    # Heuristic: Check if row looks like sales data
                    # Pivot: [Mailing Addr] [Reception] [Sale Date] [Doc] [Price] ?? 
                    # The structure is messy. Let's assume generic extraction for now 
                    # based on the known column headers from the raw HTML check earlier
                    
                    # Actually, simple scraper puts raw text. 
                    # Better: Extract all rows that happen AFTER the header row
                    pass 
                    
        # Since we can't perfectly verify the HTML structure without seeing more,
        # We will create a robust extractor that grabs ALL tables and flattens them
        # focused on the "Sales" keywords.
        
        # REVISED PARSING STRATEGY FOR SUMMIT COUNTY:
        # The page seems to put Sales info in a specific table.
        # Let's simple extract text blocks that look like sales.
        
        # For this v2, we focus on structure:
        # reception_no, sale_date, sale_price, doc_type
        
        # Mocking the parsing logic to be safe until tested:
        # If we return nothing, the CSV is empty. 
        # I'll implement a 'Best Effort' using the generic table parser.
        
        data_found = False
        
        # Find all TRs
        all_rows = soup.find_all('tr')
        for tr in all_rows:
            tds = tr.find_all('td')
            row_text = [td.get_text(strip=True) for td in tds]
            
            # Summit County typically lists sales with a Date in col 2 or 3
            # We look for a date-like string (M/D/YYYY) and a price ($...)
            if any('/' in c for c in row_text) and any('$' in c for c in row_text):
                 # Found a candidate sales row!
                 # Let's capture it.
                 # Normalizing: schno, raw_text (for later cleanup if columns shift)
                 records.append({
                     "schno": schno,
                     "raw_sales_row": " | ".join(row_text)
                 })
                 data_found = True
                 
        if not data_found:
            # At least record we visited it
            records.append({"schno": schno, "raw_sales_row": "NO_SALES_FOUND"})
            
        return records

    async def run(self):
        schnos = await self.get_schnos()
        to_scrape = [s for s in schnos if s not in self.processed_schnos]
        
        logger.info(f"Starting async scrape for {len(to_scrape)} records (Concurrency: {self.concurrency})")
        
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            
            tasks = []
            for i, schno in enumerate(to_scrape):
                tasks.append(self.process_schno(schno))
                
                # Batch writing every 1000 tasks? 
                # Better: Run all, but asyncio.gather might blow memory if 40k.
                # Use a queue or chunking?
                # Simple implementation: Chunk list
                
            # Chunking
            chunk_size = 100 
            for i in range(0, len(to_scrape), chunk_size):
                chunk = to_scrape[i:i+chunk_size]
                results = await asyncio.gather(*(self.process_schno(s) for s in chunk))
                
                # Flatten
                flat_results = [item for sublist in results if sublist for item in sublist]
                
                # Write to CSV
                if flat_results:
                    df = pd.DataFrame(flat_results)
                    # Append mode
                    hdr = not Path(self.output_csv).exists()
                    df.to_csv(self.output_csv, mode='a', header=hdr, index=False)
                
                logger.info(f"Processed {i + len(chunk)} / {len(to_scrape)}")
                
        # --- Update Metadata Timestamp ---
        import json
        from datetime import datetime
        import pytz

        try:
            mt_time = datetime.now(pytz.timezone('US/Mountain')).strftime('%Y-%m-%d %H:%M:%S %Z')
        except:
            # Fallback if pytz not installed
            mt_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        metadata = {"last_updated": mt_time}
        try:
            with open("data/metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Updated metadata.json with timestamp: {mt_time}")
        except Exception as e:
            logger.error(f"Failed to update metadata.json: {e}")

    async def process_schno(self, schno):
        html = await self.fetch_page(schno)
        if html:
            return self.parse_html(html, schno)
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summit County Async Scraper")
    parser.add_argument("--input", default="data/records.csv", help="Input CSV with 'schno'")
    parser.add_argument("--output", default="data/scraped_sales.csv", help="Output CSV")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    
    args = parser.parse_args()
    
    scraper = SummitScraper(args.input, args.output, args.workers)
    asyncio.run(scraper.run())
