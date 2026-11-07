class QuantileBoostingByCategory:
    """The other way to split the work: a separate set of 3 models (EV, P10, P90) for each category and TOTAL, each
    trained only on its own category. The production placeholder (QuantileBoosting) has one set for all categories,
    with the category as an input. A category the training data did not have uses the shared set."""
    name = "quantile_boosting_by_category"
    params = dict(QuantileBoosting.params)

    def fit(self, daily, clients, cutoff):
        wide = monthly_matrix(daily, cutoff)
        categories = wide.index.get_level_values("category")
        self.shared = QuantileBoosting().fit_months(wide)
        self.models = {c: QuantileBoosting().fit_months(wide[categories == c]) for c in sorted(set(categories))}
        return self

    def predict(self, daily, clients, cutoff):
        wide = monthly_matrix(daily, cutoff)
        categories = wide.index.get_level_values("category")
        return pd.concat([self.models.get(c, self.shared).predict_months(wide[categories == c])
                          for c in sorted(set(categories))], ignore_index=True)
