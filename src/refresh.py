import os
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.scraper import CheckeeScraper


def refresh_data(
    data_path: str = "data/raw_data.csv",
    cache_dir: str = "cache",
    months_back: int = 18,
    recent_window_days: int = 90,
) -> int:
    """
    Refresh raw_data.csv by scraping checkee.info.

    Returns the number of unique cases saved.
    """
    scraper = CheckeeScraper(cache_dir=cache_dir)

    now = datetime.now()
    months_to_fetch = []
    for i in range(months_back, -1, -1):
        d = now - relativedelta(months=i)
        months_to_fetch.append((d.year, d.month))

    all_data = []
    for year, month in months_to_fetch:
        all_data.extend(scraper.get_monthly_data(year, month))

    all_data.extend(scraper.get_recent_clears(recent_window_days))

    df = pd.DataFrame(all_data).drop_duplicates(subset=["case_id"])
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)

    return len(df)
