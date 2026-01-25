import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import datetime

class CheckeeScraper:
    BASE_URL = "https://www.checkee.info/main.php"
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_url(self, params, cache_key):
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.html")
        
        # Check cache (1 day TTL for complete cases, long for monthly)
        if os.path.exists(cache_path):
            file_mtime = os.path.getmtime(cache_path)
            if time.time() - file_mtime < 86400 or "-" in cache_key: # Cache monthly indefinitely, recent 1 day
                with open(cache_path, "r", encoding="utf-8", errors='ignore') as f:
                    return f.read()
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        print(f"Fetching from {self.BASE_URL} with {params}...")
        response = requests.get(self.BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        
        time.sleep(1.5) # Polite scraping
        
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        return response.text

    def parse_html(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        # Find all tables with border="1"
        tables = soup.find_all("table", {"border": "1"})
        
        data_table = None
        header_map = {}
        for t in tables:
            rows = t.find_all("tr")
            if not rows: continue
            headers = [td.get_text(strip=True).lower() for td in rows[0].find_all("td")]
            if "visa type" in headers and "check date" in headers:
                data_table = t
                header_map = {h: i for i, h in enumerate(headers)}
                # Handle potential duplicate headers or slightly different naming
                # The visual shows: Update, ID, Visa Type, Visa Entry, US Consulate, Major, Status, Check Date, Complete Date, Waiting Day(s), Details
                break
        
        if not data_table:
            return []
            
        rows = data_table.find_all("tr")[1:]  # Skip header
        data = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 8: # Relaxed slightly for safety
                try:
                    # Extract true numeric casenum from the Update column link (index 0)
                    update_cell = cells[header_map.get("update", 0)]
                    link = update_cell.find("a", href=True)
                    casenum = None
                    if link:
                        import re
                        m = re.search(r"casenum=(\d+)", link['href'])
                        if m:
                            casenum = int(m.group(1))
                    
                    if casenum: # Only add if we have a valid numeric ID
                        entry = {
                            "case_id": casenum,
                            "user_handle": cells[header_map.get("id", 1)].get_text(strip=True), # Keep original handle
                            "visa_type": cells[header_map.get("visa type", 2)].get_text(strip=True),
                            "visa_entry": cells[header_map.get("visa entry", 3)].get_text(strip=True),
                            "consulate": cells[header_map.get("us consulate", 4)].get_text(strip=True),
                            "major": cells[header_map.get("major", 5)].get_text(strip=True),
                            "status": cells[header_map.get("status", 6)].get_text(strip=True),
                            "check_date": cells[header_map.get("check date", 7)].get_text(strip=True),
                            "complete_date": cells[header_map.get("complete date", 8)].get_text(strip=True),
                            "waiting_days": cells[header_map.get("waiting day(s)", 9)].get_text(strip=True) if "waiting day(s)" in header_map else ""
                        }
                        data.append(entry)
                except (IndexError, KeyError):
                    continue
        return data

    def get_monthly_data(self, year, month):
        date_str = f"{year}-{month:02d}"
        html = self.fetch_url({"dispdate": date_str}, date_str)
        return self.parse_html(html)

    def get_recent_clears(self, window_days=90):
        # checkee uses specific strings for these windows
        mapping = {7: "Last 7 Days' Complete Cases", 30: "Last 30 Days' Complete Cases", 90: "Last 90 Days' Complete Cases"}
        val = mapping.get(window_days, "Last 90 Days' Complete Cases")
        html = self.fetch_url({"dispdate": val}, f"recent_{window_days}")
        return self.parse_html(html)

if __name__ == "__main__":
    scraper = CheckeeScraper()
    # Dynamically calculate months to fetch (last 18 months from now)
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    now = datetime.now()
    months_to_fetch = []
    for i in range(18, -1, -1):  # 18 months back to current
        d = now - relativedelta(months=i)
        months_to_fetch.append((d.year, d.month))
    
    all_data = []
    for year, m in months_to_fetch:
        print(f"Loading {year}-{m:02d}...")
        all_data.extend(scraper.get_monthly_data(year, m))
    
    # Collect recent clears for drift
    print("Loading recent 90 days clears...")
    all_data.extend(scraper.get_recent_clears(90))
    
    df = pd.DataFrame(all_data).drop_duplicates(subset=['case_id'])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Saved {len(df)} unique cases to data/raw_data.csv")
