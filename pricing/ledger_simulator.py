import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LedgerSimulator:
    def __init__(self, portfolio_config: dict, bond_pricers: dict, irs_pricers: dict, 
                 data_processor, injected_break: float = 0.0): 
        self.portfolio      = portfolio_config
        self.bond_pricers   = bond_pricers
        self.irs_pricers    = irs_pricers
        self.dp             = data_processor
        self.injected_break = injected_break 

    def _eir_book_value(self, asset_cfg: dict, settle: pd.Timestamp) -> float:
        face       = asset_cfg["face_value"]
        purchase   = asset_cfg["purchase_price"] / 100 * face
        eir        = asset_cfg["purchase_yield"]
        issue      = pd.Timestamp(asset_cfg["issue_date"])
        maturity   = pd.Timestamp(asset_cfg["maturity_date"])
        total_days = (maturity - issue).days

        # Straight-line amortisation of premium/discount
        days_elapsed = (settle - issue).days
        fraction     = min(days_elapsed / total_days, 1.0)
        book_value   = purchase + (face - purchase) * fraction
        return book_value

    def fo_value(self, asset_cfg: dict, settle: pd.Timestamp) -> float:
        aid = asset_cfg["id"]
        if asset_cfg["type"] == "bond":
            yield_ = self.dp.get_yield_for_tenor(asset_cfg["fred_tenor"], settle)
            return self.bond_pricers[aid].present_value(settle, yield_)
        elif asset_cfg["type"] == "irs":
            return self.irs_pricers[aid].npv(settle, self.dp)
        return 0.0

    def ledger_value(self, asset_cfg: dict, settle: pd.Timestamp) -> float:
        aid      = asset_cfg["id"]
        category = asset_cfg["ifrs_category"]

        if category == "amortised_cost":
            base = self._eir_book_value(asset_cfg, settle)
        else:
            # FVTPL → same as FO
            base = self.fo_value(asset_cfg, settle)

        return base

    def reconcile(self, settle: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for asset in self.portfolio["assets"]:
            fo_val  = self.fo_value(asset, settle)
            leg_val = self.ledger_value(asset, settle)
            diff    = leg_val - fo_val
            rows.append({
                "asset_id":       asset["id"],
                "ifrs_category":  asset["ifrs_category"],
                "book":           asset["book"],
                "fo_value_usd":   fo_val,
                "ledger_value_usd": leg_val,
                "break_usd":      diff,
                "break_flag":     abs(diff) > self.portfolio.get(
                                      "settings", {}).get("break_threshold_usd", 5_000),
                "settle":         settle,
            })
        return pd.DataFrame(rows)
