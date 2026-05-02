import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class IRSPricer:
    def __init__(self, asset_config: dict):
        self.cfg        = asset_config
        self.notional   = asset_config["notional"]
        self.fixed_rate = asset_config["fixed_rate"]
        self.freq       = asset_config["payment_frequency"]  # per year
        self.effective  = pd.Timestamp(asset_config["effective_date"])
        self.maturity   = pd.Timestamp(asset_config["maturity_date"])
        self.direction  = asset_config["direction"]          # "pay_fixed" | "receive_fixed"

    def _payment_dates(self, settle: pd.Timestamp) -> list[pd.Timestamp]:
        dates = []
        d = self.maturity
        while d > settle:
            dates.append(d)
            d -= relativedelta(months=int(12 / self.freq))
        return sorted(dates)

    def _accrual_fraction(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        return (end - start).days / 360  # ACT/360 for USD IRS

    def _discount_factor(self, settle: pd.Timestamp, pay_date: pd.Timestamp,
                         data_processor, rate_curve: dict) -> float:
        years = (pay_date - settle).days / 365.25
        if years <= 0:
            return 1.0
        # Interpolate from data_processor
        r = data_processor.interpolate_yield(settle, years)
        return np.exp(-r * years)  # Continuous compounding discount

    def fixed_leg_pv(self, settle: pd.Timestamp, data_processor) -> float:
        pay_dates = self._payment_dates(settle)
        prev = max(self.effective, settle)
        pv   = 0.0
        for pd_ in pay_dates:
            accrual = self._accrual_fraction(prev, pd_)
            cf      = self.fixed_rate * accrual * self.notional
            df      = self._discount_factor(settle, pd_, data_processor, {})
            pv     += cf * df
            prev    = pd_
        return pv

    def floating_leg_pv(self, settle: pd.Timestamp, data_processor) -> float:
        pay_dates = self._payment_dates(settle)
        prev = max(self.effective, settle)
        pv = 0.0
        for pd_ in pay_dates:
            t_start = (prev - settle).days / 365.25
            t_end   = (pd_ - settle).days / 365.25
            r_start = data_processor.interpolate_yield(settle, max(t_start, 1e-6))
            r_end   = data_processor.interpolate_yield(settle, t_end)
            df_start = np.exp(-r_start * t_start) if t_start > 0 else 1.0
            df_end   = np.exp(-r_end * t_end)
            # Use same 365.25 basis as discount factors for forward extraction
            t_accrual = (pd_ - prev).days / 365.25   
            forward_rate = (df_start / df_end - 1) / t_accrual
            accrual = self._accrual_fraction(prev, pd_)  # ACT/360 for cash flow sizing
            cf = forward_rate * accrual * self.notional
            pv += cf * df_end
            prev = pd_
        return pv

    def npv(self, settle: pd.Timestamp, data_processor) -> float:
        float_pv = self.floating_leg_pv(settle, data_processor)
        fixed_pv = self.fixed_leg_pv(settle, data_processor)
        raw_npv  = float_pv - fixed_pv   # Pay fixed → receive float − pay fixed
        return raw_npv if self.direction == "pay_fixed" else -raw_npv

    # ------------------------------------------------------------------
    # DV01 of the swap
    # ------------------------------------------------------------------
    def dv01(self, settle: pd.Timestamp, data_processor, bump_bps: float = 1.0) -> float:
        # Bump SOFR rate by +1bp for DV01 approximation on floating leg
        sofr_orig = data_processor.get_sofr(settle)

        class BumpedProcessor:
            def __init__(self, base, bump):
                self._base = base
                self._bump = bump
            def get_sofr(self, d):
                return self._base.get_sofr(d) + self._bump
            def interpolate_yield(self, d, t):
                return self._base.interpolate_yield(d, t) + self._bump
            def get_yield_for_tenor(self, tk, d):
                return self._base.get_yield_for_tenor(tk, d) + self._bump

        bump    = bump_bps / 10_000
        bp_up   = BumpedProcessor(data_processor,  bump)
        bp_down = BumpedProcessor(data_processor, -bump)
        return (self.npv(settle, bp_up) - self.npv(settle, bp_down)) / 2

    # ------------------------------------------------------------------
    # Daily carry (net coupon accrual)
    # ------------------------------------------------------------------
    def daily_carry(self, settle: pd.Timestamp, data_processor) -> float:
        sofr  = data_processor.get_sofr(settle)
        daily = (sofr - self.fixed_rate) * self.notional / 360
        return daily if self.direction == "pay_fixed" else -daily

    def price_summary(self, settle: pd.Timestamp, data_processor) -> dict:
        return {
            "asset_id":    self.cfg["id"],
            "settle":      settle,
            "fixed_leg_pv": self.fixed_leg_pv(settle, data_processor),
            "float_leg_pv": self.floating_leg_pv(settle, data_processor),
            "npv_usd":     self.npv(settle, data_processor),
            "dv01_usd":    self.dv01(settle, data_processor),
            "daily_carry": self.daily_carry(settle, data_processor),
        }
