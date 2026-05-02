import logging
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

TENOR_YEARS = {
    "1M":  1/12,
    "3M":  3/12,
    "6M":  6/12,
    "1Y":  1.0,
    "2Y":  2.0,
    "5Y":  5.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}


class DataProcessor:
    def __init__(self, yield_curve: pd.DataFrame, sofr: pd.Series):
        self.raw_yield_curve = yield_curve.copy()
        self.raw_sofr        = sofr.copy()
        self.yield_curve     = self._clean_yield_curve(yield_curve)
        self.sofr            = self._clean_sofr(sofr)

    def _clean_yield_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.ffill(inplace=True)          # Forward-fill weekends/holidays
        df.dropna(how="all", inplace=True)
        # Sanity clip: yields should be between -2% and 20%
        df = df.clip(-0.02, 0.20)
        return df

    def _clean_sofr(self, s: pd.Series) -> pd.Series:
        s = s.copy()
        s.ffill(inplace=True)
        s = s.clip(-0.02, 0.20)
        return s

    def interpolate_yield(self, date: pd.Timestamp, target_tenor_years: float) -> float:
        """Cubic spline interpolation for any tenor on a given date."""
        row = self.yield_curve.loc[:date].iloc[-1]
        tenors = []
        yields = []
        for col, yr in TENOR_YEARS.items():
            if col in row.index and not np.isnan(row[col]):
                tenors.append(yr)
                yields.append(row[col])
        if len(tenors) < 2:
            raise ValueError(f"Not enough tenor points for interpolation on {date.date()}")
        cs = CubicSpline(tenors, yields)
        return float(cs(target_tenor_years))

    def get_yield_for_tenor(self, tenor_key: str, date: pd.Timestamp) -> float:
        """Fetch a named tenor yield (e.g. '10Y') for a specific date."""
        try:
            row = self.yield_curve.loc[:date].iloc[-1]
            return float(row[tenor_key])
        except (KeyError, IndexError):
            years = TENOR_YEARS[tenor_key]
            return self.interpolate_yield(date, years)

    def get_sofr(self, date: pd.Timestamp) -> float:
        sub = self.sofr.loc[:date]
        if sub.empty:
            raise ValueError(f"No SOFR data available on or before {date.date()}")
        return float(sub.iloc[-1])

    def latest_date(self) -> pd.Timestamp:
        return self.yield_curve.index[-1]

    def previous_business_date(self, date: pd.Timestamp) -> pd.Timestamp:
        idx = self.yield_curve.index
        prior = idx[idx < date]
        if prior.empty:
            raise ValueError(f"No prior business date available before {date.date()}")
        return prior[-1]

    def yield_change(self, tenor_key: str, date_t: pd.Timestamp, date_t1: pd.Timestamp) -> float:
        """Returns yield change in decimal (not bps): y_T - y_{T-1}"""
        return self.get_yield_for_tenor(tenor_key, date_t) - self.get_yield_for_tenor(tenor_key, date_t1)