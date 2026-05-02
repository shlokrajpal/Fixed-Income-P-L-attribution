import os
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

YIELD_CURVE_SERIES = {
    "1M":  "DGS1MO",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}
SOFR_SERIES = "SOFR"


class FredClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError("FRED_API_KEY not set. Export it as an environment variable or pass directly.")
        self.fred = Fred(api_key=key)

    def _date_range(self, window_days: int = 30):
        end   = datetime.today()
        start = end - timedelta(days=window_days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def fetch_yield_curve(self, window_days: int = 30) -> pd.DataFrame:
        start, end = self._date_range(window_days)
        frames = {}
        for tenor, series_id in YIELD_CURVE_SERIES.items():
            try:
                s = self.fred.get_series(series_id, observation_start=start, observation_end=end)
                frames[tenor] = s / 100  # Convert % to decimal
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.dropna(how="all", inplace=True)
        logger.info(f"Yield curve fetched: {len(df)} rows, {df.index[-1].date()} latest date")
        return df

    def fetch_sofr(self, window_days: int = 30) -> pd.Series:
        start, end = self._date_range(window_days)
        try:
            s = self.fred.get_series(SOFR_SERIES, observation_start=start, observation_end=end)
            s = s / 100
            s.index = pd.to_datetime(s.index)
            s.name = "SOFR"
            logger.info(f"SOFR fetched: latest = {s.iloc[-1]:.4%} on {s.index[-1].date()}")
            return s
        except Exception as e:
            logger.error(f"Failed to fetch SOFR: {e}")
            raise

    def fetch_all(self, window_days: int = 30) -> dict:
        yc   = self.fetch_yield_curve(window_days)
        sofr = self.fetch_sofr(window_days)
        # Align SOFR to yield curve dates — SOFR publishes T+1, trim the excess
        sofr = sofr[sofr.index <= yc.index.max()]
        return {"yield_curve": yc, "sofr": sofr}
