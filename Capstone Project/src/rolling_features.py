def add_rolling_features(df, windows=(7, 28), col="sales", group="id"):
    """Rolling mean/std of PAST sales. shift(1) first to exclude the current
    day (prevents leakage), then roll. Assumes df sorted by [id, date]."""
    df = df.copy()
    # shift(1) once: every rolling stat is computed on data up to yesterday
    shifted = df.groupby(group, observed=True)[col].shift(1)
    for w in windows:
        roll = shifted.rolling(w)
        df[f"roll_mean_{w}"] = roll.mean().astype("float32")
        df[f"roll_std_{w}"]  = roll.std().astype("float32")
    return df