import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd


class BondPricer:
    def __init__(self, asset_config: dict):
        self.cfg            = asset_config
        self.face           = asset_config["face_value"]
        self.coupon_rate    = asset_config["coupon_rate"]
        self.freq           = asset_config["coupon_frequency"]
        self.maturity       = pd.Timestamp(asset_config["maturity_date"])
        self.issue          = pd.Timestamp(asset_config["issue_date"])
        self.coupon_payment = (self.coupon_rate / self.freq) * self.face

    # ------------------------------------------------------------------
    # Cash flow schedule
    # ------------------------------------------------------------------
    def _coupon_dates(self, settle: pd.Timestamp) -> list[pd.Timestamp]:
        dates = []
        d = self.maturity
        while d > settle:
            dates.append(d)
            d -= relativedelta(months=int(12 / self.freq))
        return sorted(dates)

    def _time_to_cashflow(self, cf_date: pd.Timestamp, settle: pd.Timestamp) -> float:
        return (cf_date - settle).days / 365.25

    # ------------------------------------------------------------------
    # Dirty price (full price) as % of par
    # ------------------------------------------------------------------
    def dirty_price(self, settle: pd.Timestamp, yield_: float) -> float:
        cf_dates = self._coupon_dates(settle)
        pv = 0.0
        for cf_date in cf_dates:
            t  = self._time_to_cashflow(cf_date, settle)
            cf = self.coupon_payment
            if cf_date == self.maturity:
                cf += self.face
            pv += cf / ((1 + yield_ / self.freq) ** (t * self.freq))
        return (pv / self.face) * 100

    # ------------------------------------------------------------------
    # Accrued interest
    # ------------------------------------------------------------------
    def accrued_interest(self, settle: pd.Timestamp) -> float:
        cf_dates = self._coupon_dates(settle)
        prev_coupon = self.issue
        for cd in cf_dates:
            if cd <= settle:
                prev_coupon = cd
                break
        days_accrued = (settle - prev_coupon).days
        period_days  = 365.25 / self.freq
        accrued_pct  = (days_accrued / period_days) * (self.coupon_rate / self.freq) * 100
        return accrued_pct * self.face / 100

    # ------------------------------------------------------------------
    # Clean price
    # ------------------------------------------------------------------
    def clean_price(self, settle: pd.Timestamp, yield_: float) -> float:
        dp  = self.dirty_price(settle, yield_)
        ai  = (self.accrued_interest(settle) / self.face) * 100
        return dp - ai

    # ------------------------------------------------------------------
    # Full PV in USD
    # ------------------------------------------------------------------
    def present_value(self, settle: pd.Timestamp, yield_: float) -> float:
        return (self.dirty_price(settle, yield_) / 100) * self.face

    # ------------------------------------------------------------------
    # DV01 — price sensitivity to 1bp yield move
    # ------------------------------------------------------------------
    def dv01(self, settle: pd.Timestamp, yield_: float, bump_bps: float = 1.0) -> float:
        bump    = bump_bps / 10_000
        pv_up   = self.present_value(settle, yield_ + bump)
        pv_down = self.present_value(settle, yield_ - bump)
        return (pv_down - pv_up) / 2  # Dollar DV01 (positive = long)

    # ------------------------------------------------------------------
    # Daily carry (accrual)
    # ------------------------------------------------------------------
    def daily_carry(self) -> float:
        return (self.coupon_rate * self.face) / 365.25

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------
    def price_summary(self, settle: pd.Timestamp, yield_: float) -> dict:
        return {
            "asset_id":       self.cfg["id"],
            "settle":         settle,
            "yield":          yield_,
            "dirty_price_pct": self.dirty_price(settle, yield_),
            "clean_price_pct": self.clean_price(settle, yield_),
            "accrued_usd":    self.accrued_interest(settle),
            "pv_usd":         self.present_value(settle, yield_),
            "dv01_usd":       self.dv01(settle, yield_),
            "daily_carry":    self.daily_carry(),
        }
