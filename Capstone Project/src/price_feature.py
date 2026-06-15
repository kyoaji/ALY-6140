def add_price_features(df, group="id", window=56, min_periods=14):
    """Create a price comparison feature.

    price_vs_median shows whether the current price is higher or lower than
    the recent normal price. The previous prices are used to avoid leakage.
    """
    df = df.copy()
    g = df.groupby(group, observed=True)["sell_price"]
    recent_median = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).median()
    )
    df["price_vs_median"] = (df["sell_price"] / recent_median).astype("float32")
    return df