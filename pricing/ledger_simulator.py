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

        days_elapsed = (settle - issue).days
        fraction     = min(days_elapsed / total_days, 1.0)
        book_value   = purchase + (face - purchase) * fraction
        return book_value

    def fo_value(self, asset_cfg: dict, settle: pd.Timestamp) -> float:
        aid = asset_cfg["id"]
        if asset_cfg["type"] == "bond":
            pricer = self.bond_pricers[asset_cfg["id"]]
            ttm = (pricer.maturity - settle).days / 365.25
            yield_ = self.dp.interpolate_yield(settle, max(ttm, 1e-6))
            return pricer.present_value(settle, yield_)
        elif asset_cfg["type"] == "irs":
            return self.irs_pricers[aid].npv(settle, self.dp)
        return 0.0

    def ledger_value(self, asset_cfg: dict, settle: pd.Timestamp) -> float:
        category = asset_cfg["ifrs_category"]
        if category == "amortised_cost":
            return self._eir_book_value(asset_cfg, settle)
        else:
            # FVTPL → ledger should match FO mark-to-market
            return self.fo_value(asset_cfg, settle)

    def reconcile(self, settle: pd.Timestamp) -> pd.DataFrame:
        threshold = self.portfolio.get("settings", {}).get("break_threshold_usd", 5_000)
        rows = []
        for asset in self.portfolio["assets"]:
            fo_val  = self.fo_value(asset, settle)
            leg_val = self.ledger_value(asset, settle)

            # Amortised Cost divergence from MTM is expected — not a break.
            # Track it as unrealized P&L for informational purposes only.
            if asset["ifrs_category"] == "amortised_cost":
                diff               = 0.0
                unrealized_gain_loss = fo_val - leg_val   # MTM premium/discount vs book
            else:
                # FVTPL: ledger and FO should agree; any gap is a genuine operational break
                diff               = leg_val - fo_val
                unrealized_gain_loss = 0.0

            rows.append({
                "asset_id":              asset["id"],
                "ifrs_category":         asset["ifrs_category"],
                "book":                  asset["book"],
                "fo_value_usd":          fo_val,
                "ledger_value_usd":      leg_val,
                "unrealized_gain_loss":  unrealized_gain_loss,
                "break_usd":             diff,
                "break_flag":            abs(diff) > threshold,
                "settle":                settle,
            })
        return pd.DataFrame(rows)