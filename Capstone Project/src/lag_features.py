def add_lag_features(df, lags=(7, 14, 28), col="sales", group="id"):
    """Shifted past sales. groupby(id) prevents cross-series leakage;
    shift(N) only looks backward. Assumes df sorted by [id, date]."""
    df = df.copy()
    grouped = df.groupby(group, observed=True)[col]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag).astype("float32")
    return df