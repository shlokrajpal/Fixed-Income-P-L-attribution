import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

EXCEPTION_THRESHOLD = 500   # USD — residual above this flags a model exception


class PLDecomposer:
    def __init__(self, portfolio_config: dict, bond_pricers: dict, irs_pricers: dict,
                 data_processor):
        self.portfolio    = portfolio_config
        self.bond_pricers = bond_pricers
        self.irs_pricers  = irs_pricers
        self.dp           = data_processor

    def _decompose_bond(self, asset: dict, date_t: pd.Timestamp, date_t1: pd.Timestamp) -> dict:
        aid    = asset["id"]
        pricer = self.bond_pricers[aid]

        # Dynamically compute TTM and interpolate yield — no more hardcoded fred_tenor
        ttm_years = (pricer.maturity - date_t).days / 365.25
        if ttm_years <= 0:
            return self._build_row(asset, date_t, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0)

        y_t  = self.dp.interpolate_yield(date_t,  ttm_years)
        y_t1 = self.dp.interpolate_yield(date_t1, ttm_years)

        pv_t  = pricer.present_value(date_t,  y_t)
        pv_t1 = pricer.present_value(date_t1, y_t1)

        total_pl    = pv_t - pv_t1
        dv01_t      = pricer.dv01(date_t,  y_t)
        dv01_t1     = pricer.dv01(date_t1, y_t1)
        dv01_       = (dv01_t + dv01_t1) / 2          # Average for convexity
        delta_y_bps = (y_t - y_t1) * 10_000
        rate_move   = -(dv01_ * delta_y_bps)
        carry       = pricer.daily_carry(date_t)       # pass settle date
        residual    = total_pl - (rate_move + carry)

        return self._build_row(asset, date_t, total_pl, rate_move, carry, residual,
                               y_t, y_t1, delta_y_bps, dv01_)

    def _decompose_irs(self, asset: dict, date_t: pd.Timestamp, date_t1: pd.Timestamp) -> dict:
        aid    = asset["id"]
        pricer = self.irs_pricers[aid]

        npv_t  = pricer.npv(date_t,  self.dp)
        npv_t1 = pricer.npv(date_t1, self.dp)
        total_pl = npv_t - npv_t1

        # Dynamic TTM — swap shortens daily, interpolate at true remaining maturity
        ttm_years = (pricer.maturity - date_t).days / 365.25
        if ttm_years <= 0:
            return self._build_row(asset, date_t, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0)

        # Average DV01 to account for convexity over the day
        dv01_t  = pricer.dv01(date_t,  self.dp)
        dv01_t1 = pricer.dv01(date_t1, self.dp)
        dv01_   = (dv01_t + dv01_t1) / 2

        # Interpolate yield at the swap's actual remaining tenor on both dates
        y_t  = self.dp.interpolate_yield(date_t,  ttm_years)
        y_t1 = self.dp.interpolate_yield(date_t1, ttm_years)

        delta_y_bps = (y_t - y_t1) * 10_000
        rate_move   = dv01_ * delta_y_bps

        carry    = pricer.daily_carry(date_t1, self.dp)
        residual = total_pl - (rate_move + carry)

        return self._build_row(asset, date_t, total_pl, rate_move, carry, residual,
                               y_t, y_t1, delta_y_bps, dv01_)

    def _build_row(self, asset, date_t, total_pl, rate_move, carry, residual,
                   y_t, y_t1, delta_y_bps, dv01_) -> dict:
        return {
            "asset_id":        asset["id"],
            "ifrs_category":   asset["ifrs_category"],
            "book":            asset["book"],
            "date":            date_t,
            "total_pl_usd":    round(total_pl,    2),
            "rate_move_usd":   round(rate_move,   2),
            "carry_usd":       round(carry,        2),
            "residual_usd":    round(residual,     2),
            "yield_t":         round(y_t,          6),
            "yield_t1":        round(y_t1,         6),
            "delta_yield_bps": round(delta_y_bps,  4),
            "dv01_usd":        round(dv01_,        2),
            "model_exception": abs(residual) > EXCEPTION_THRESHOLD,
        }

    def decompose(self, date_t: pd.Timestamp) -> pd.DataFrame:
        date_t1 = self.dp.previous_business_date(date_t)
        logger.info(f"Decomposing P&L: T={date_t.date()}, T-1={date_t1.date()}")

        from pricing.ledger_simulator import LedgerSimulator
        ledger = LedgerSimulator(self.portfolio, self.bond_pricers, self.irs_pricers, self.dp)

        rows = []
        for asset in self.portfolio["assets"]:
            try:
                if asset["ifrs_category"] == "amortised_cost":
                    val_t  = ledger.ledger_value(asset, date_t)
                    val_t1 = ledger.ledger_value(asset, date_t1)
                    total_pl = val_t - val_t1
                    row = self._build_row(
                        asset, date_t, total_pl,
                        rate_move=0.0, carry=total_pl, residual=0.0,
                        y_t=0.0, y_t1=0.0, delta_y_bps=0.0, dv01_=0.0
                    )
                elif asset["type"] == "bond":
                    row = self._decompose_bond(asset, date_t, date_t1)
                elif asset["type"] == "irs":
                    row = self._decompose_irs(asset, date_t, date_t1)
                else:
                    continue
                rows.append(row)
            except Exception as e:
                logger.error(f"Failed to decompose {asset['id']}: {e}")

        df = pd.DataFrame(rows)
        return self._apply_totals(df, date_t)

    def _apply_totals(self, df, date_t):
        if df.empty:
            return df
        totals = {
            "asset_id":        "TOTAL",
            "total_pl_usd":    df["total_pl_usd"].sum(),
            "rate_move_usd":   df["rate_move_usd"].sum(),
            "carry_usd":       df["carry_usd"].sum(),
            "residual_usd":    df["residual_usd"].sum(),
            "dv01_usd":        df["dv01_usd"].sum(),
            "model_exception": df["model_exception"].any(),
            "date":            date_t,
            "ifrs_category":   "",
            "book":            "",
            "yield_t":         np.nan,
            "yield_t1":        np.nan,
            "delta_yield_bps": np.nan,
        }
        return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)