class SimpleGuess:
    """The simple guess as a forecaster: EV = average of the last 3 months (without the scheduled part); P10 and P90 =
    EV plus the errors it made in training at 10% and 90%, per category, scaled by the series scale."""
    name = "simple_guess"
    params = {"quantiles": "0.1,0.9"}
    inputs = ["category", "free_avg_3m", "series_scale"]

    def _guess(self, rows):
        return np.where(rows["category"] == "NET", rows["free_avg_3m"], rows["free_avg_3m"].clip(lower=0))

    def fit(self, rows):
        error = (rows["target"] - self._guess(rows)) / rows["series_scale"]
        self.bands = error.groupby(rows["category"]).quantile([0.1, 0.9]).unstack()

    def predict(self, rows):
        ev = self._guess(rows)
        band = self.bands.reindex(rows["category"]).fillna(0.0).values
        scale = rows["series_scale"].values
        return pd.DataFrame({"p10": ev + band[:, 0] * scale, "ev": ev, "p90": ev + band[:, 1] * scale},
                            index=rows.index)


class CatBoostDepth8(CatBoostForecaster):
    """The reference CatBoost forecaster with deeper trees."""
    name = "catboost_depth8"
    params = {**CatBoostForecaster.params, "depth": 8}


class SklearnBoosting:
    """scikit-learn gradient boosting (HistGradientBoostingRegressor): one model for the EV, one for P10 and one for
    P90 (quantile loss), with the reference forecaster's inputs and scaling (amounts / series scale)."""
    name = "sklearn_boosting"
    params = {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40, "random_state": 0}
    inputs = CatBoostForecaster.inputs

    def _matrix(self, rows):
        X = model_matrix(rows)
        for c in CAT_FEATURES:            # categories as numbers; one not seen in training counts as missing
            X[c] = X[c].map(self.codes[c]).astype(float)
        return X

    def fit(self, rows):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.codes = {c: {v: i for i, v in enumerate(sorted(values.unique()))}
                      for c, values in model_matrix(rows)[CAT_FEATURES].items()}
        X, y = self._matrix(rows), rows["target"] / rows["series_scale"]
        is_category = [c in CAT_FEATURES for c in X.columns]
        losses = {"ev": {"loss": "squared_error"}, "p10": {"loss": "quantile", "quantile": 0.1},
                  "p90": {"loss": "quantile", "quantile": 0.9}}
        self.models = {k: HistGradientBoostingRegressor(categorical_features=is_category, **loss, **self.params).fit(X, y)
                       for k, loss in losses.items()}

    def predict(self, rows):
        X, scale = self._matrix(rows), rows["series_scale"].values
        return pd.DataFrame({k: m.predict(X) * scale for k, m in self.models.items()}, index=rows.index)
