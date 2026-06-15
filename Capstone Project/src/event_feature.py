def add_event_features(df):
    """Encode primary event (name + type). Secondary events dropped:
    99.8% missing, negligible info."""
    df = df.copy()
    for c in ["event_name_1", "event_type_1"]:
        df[c] = df[c].astype("object").fillna("none").astype("category")
    df = df.drop(columns=["event_name_2", "event_type_2"], errors="ignore")
    return df