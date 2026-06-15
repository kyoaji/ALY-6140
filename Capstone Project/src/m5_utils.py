def optimize_dtypes(df):
    """
    Downcasts columns of the M5 long DataFrame to compact data types
    to significantly reduce memory footprint.
    """
    df = df.copy()

    # 1. Sales Volume
    if "sales" in df:
        df["sales"] = df["sales"].astype("int16")

    # 2. Small-range integers (Calendar features)
    int8_cols = ["wday", "month"]
    for c in int8_cols:
        if c in df:
            df[c] = df[c].astype("int8")

    # Catching the binary SNAP columns (0 or 1)
    snap_cols = [c for c in df.columns if c.startswith("snap_")]
    for c in snap_cols:
        df[c] = df[c].astype("int8")

    if "year" in df:
        df["year"] = df["year"].astype("int16")
    if "wm_yr_wk" in df:
        df["wm_yr_wk"] = df["wm_yr_wk"].astype("int32")
    if "d_num" in df:
        df["d_num"] = df["d_num"].astype("int32")

    # 3. Prices
    if "sell_price" in df:
        df["sell_price"] = df["sell_price"].astype("float32")

    # 4. Identifiers & Event columns (Added "d" to this list)
    cat_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for c in cat_cols:
        if c in df:
            df[c] = df[c].astype("category")

    return df