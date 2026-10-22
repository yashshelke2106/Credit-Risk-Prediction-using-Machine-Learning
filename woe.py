"""Weight-of-Evidence binning as a scikit-learn transformer.

Written as a transformer on purpose: bin edges and WoE values are FITTED
quantities derived from the target, so computing them on the full dataset and
then cross-validating is exactly the leak this project is being audited for.
As a transformer it is refit inside every CV fold.

Convention: WoE = ln(%good / %bad), with bad == loan_status 1 (default).
Higher WoE therefore means lower risk, so scorecard points rise with WoE.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SMOOTH = 0.5          # Laplace term; stops WoE exploding when a bin has 0 bads
MISSING = "Missing"
OTHER = "Other"


class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, min_bin_frac=0.05, max_cat_levels=10):
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.max_cat_levels = max_cat_levels

    # ---------------------------------------------------------------- fitting
    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(np.asarray(y), index=X.index)
        self.numeric_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_ = [c for c in X.columns if c not in self.numeric_]
        self.edges_, self.woe_maps_, self.iv_, self.bin_tables_ = {}, {}, {}, {}

        tot_good = float((y == 0).sum())
        tot_bad = float((y == 1).sum())

        for col in X.columns:
            if col in self.numeric_:
                s = X[col]
                finite = s.dropna()
                # Quantile edges on the TRAINING fold only.
                qs = np.linspace(0, 1, self.n_bins + 1)
                edges = np.unique(np.quantile(finite, qs))
                if len(edges) < 3:          # near-constant column
                    edges = np.unique([finite.min(), finite.max()])
                edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
                self.edges_[col] = edges
                binned = pd.cut(s, bins=edges, duplicates="drop")
                binned = binned.cat.add_categories([MISSING]).fillna(MISSING)
            else:
                s = X[col].astype("object").where(X[col].notna(), MISSING)
                keep = s.value_counts().head(self.max_cat_levels).index
                binned = s.where(s.isin(keep), OTHER)
                self.edges_[col] = list(keep)

            tab = self._woe_table(binned, y, tot_good, tot_bad)
            self.woe_maps_[col] = tab["WoE"].to_dict()
            self.iv_[col] = float(tab["IV_contrib"].sum())
            self.bin_tables_[col] = tab
        return self

    @staticmethod
    def _woe_table(binned, y, tot_good, tot_bad):
        df = pd.DataFrame({"bin": binned.astype(str), "y": y.values})
        g = df.groupby("bin", observed=True)["y"].agg(Count="size", Bads="sum")
        g["Goods"] = g["Count"] - g["Bads"]
        g["DistGood"] = (g["Goods"] + SMOOTH) / (tot_good + SMOOTH * len(g))
        g["DistBad"] = (g["Bads"] + SMOOTH) / (tot_bad + SMOOTH * len(g))
        g["BadRate"] = g["Bads"] / g["Count"]
        g["WoE"] = np.log(g["DistGood"] / g["DistBad"])
        g["IV_contrib"] = (g["DistGood"] - g["DistBad"]) * g["WoE"]
        return g

    # ------------------------------------------------------------- transform
    def _bin_column(self, X, col):
        if col in self.numeric_:
            b = pd.cut(X[col], bins=self.edges_[col], duplicates="drop").astype(str)
            return b.where(X[col].notna(), MISSING)
        s = X[col].astype("object").where(X[col].notna(), MISSING)
        return s.where(s.isin(self.edges_[col]) | (s == MISSING), OTHER)

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.woe_maps_:
            # Unseen bin -> WoE 0, i.e. neutral, never a guess at its risk.
            out[col + "_woe"] = self._bin_column(X, col).map(self.woe_maps_[col]).fillna(0.0).astype(float)
        return out.values

    def iv_table(self):
        return (pd.DataFrame({"Feature": list(self.iv_), "IV": list(self.iv_.values())})
                .sort_values("IV", ascending=False).reset_index(drop=True))
