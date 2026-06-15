def add_calendar_features(df):
    """Extract time attributes from date."""
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")   # 0=Mon ... 6=Sun
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")   # Sat/Sun
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    return df