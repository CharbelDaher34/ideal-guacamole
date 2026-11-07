from sklearn.ensemble import HistGradientBoostingRegressor


class LagBoosting:
    """Example of a small model that learns: scikit-learn gradient boosting on the last 12 months of each client and
    category (divided by the series' scale, so small and large clients look alike), one model each for the EV (the
    median), P10 and P90 (quantile loss). Trained on every client and category at each of the 24 cut-offs before the
    last."""
    name = "lag_boosting"
    params = {"lags": 12, "cutoffs": 24, "max_iter": 200, "learning_rate": 0.05}

    def _rows(self, wide, p, h=None):
        """Inputs at month position p (the cut-off) for months ahead h; with the answer if h is known."""
        last = wide.iloc[:, max(0, p - self.params["lags"] + 1):p + 1]
        scale = last.abs().mean(axis=1).fillna(0).clip(lower=100.0)
        X = last.div(scale, axis=0).reindex(columns=last.columns).fillna(0.0)
        X.columns = [f"lag_{len(X.columns) - i}" for i in range(len(X.columns))]
        X = X.reindex(columns=[f"lag_{k}" for k in range(self.params["lags"], 0, -1)], fill_value=0.0)
        X["total"] = (wide.index.get_level_values("category") == TOTAL).astype(int)
        X["months_known"] = last.notna().sum(axis=1).values
        X["months_ahead"] = h or 0
        return X, scale

    def fit(self, daily, clients, cutoff):
        wide = monthly_matrix(daily, cutoff)
        n, Xs, ys = wide.shape[1], [], []
        for p in range(max(0, n - 1 - self.params["cutoffs"]), n - 1):
            for h in HORIZONS:
                if p + h < n:
                    X, scale = self._rows(wide, p, h)
                    y = wide.iloc[:, p + h] / scale
                    ok = y.notna() & wide.iloc[:, p].notna()
                    Xs.append(X[ok.values])
                    ys.append(y[ok])
        X, y = pd.concat(Xs), pd.concat(ys)
        settings = {"max_iter": self.params["max_iter"], "learning_rate": self.params["learning_rate"],
                    "random_state": 0}
        self.models = {"ev": HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **settings).fit(X, y),
                       "p10": HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **settings).fit(X, y),
                       "p90": HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **settings).fit(X, y)}
        return self

    def predict(self, daily, clients, cutoff):
        wide = monthly_matrix(daily, cutoff)
        out = []
        for h in HORIZONS:
            X, scale = self._rows(wide, wide.shape[1] - 1, h)
            rows = wide.index.to_frame(index=False).assign(months_ahead=h)
            for column, model in self.models.items():
                rows[column] = model.predict(X) * scale.values
            out.append(rows)
        return pd.concat(out, ignore_index=True)[FORECAST_COLUMNS]
