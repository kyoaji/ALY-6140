def recursive_forecast(model, sales_wide, price_wide, test_skeleton,
                       feature_cols, categorical_feats,
                       horizon_start, horizon_end,
                       price_window=56, price_min_periods=14):
    """Recursive multi-step forecast.
    For each test day, compute lag/rolling/price_vs_median from the running
    sales/price history, predict, write the prediction back so later days'
    lags can use it. Returns long DataFrame [id, d_num, prediction]."""
    import numpy as np
    import pandas as pd

    series_ids = sales_wide.index
    # static + known features per series, indexed for fast per-day lookup
    # test_skeleton has one row per (id, d_num); set index for slicing
    skel = test_skeleton.set_index(["d_num", "id"]).sort_index()

    preds_all = []

    for d in range(horizon_start, horizon_end + 1):
        # ---- 1. known features for day d (calendar/price/event/snap) ----
        known = skel.loc[d].reindex(series_ids)   # align to sales_wide order

        # ---- 2. lag features from running sales history ----
        feat = pd.DataFrame(index=series_ids)
        feat["lag_7"]  = sales_wide[d - 7].values
        feat["lag_14"] = sales_wide[d - 14].values
        feat["lag_28"] = sales_wide[d - 28].values

        # rolling stats over PAST 7/28 days (days d-7..d-1, d-28..d-1)
        past7  = sales_wide.loc[:, d-7:d-1]
        past28 = sales_wide.loc[:, d-28:d-1]
        feat["roll_mean_7"]  = past7.mean(axis=1).values
        feat["roll_std_7"]   = past7.std(axis=1).values
        feat["roll_mean_28"] = past28.mean(axis=1).values
        feat["roll_std_28"]  = past28.std(axis=1).values

        # ---- 3. price_vs_median: current price / rolling median of past prices ----
        cur_price = price_wide[d].values
        past_price = price_wide.loc[:, d-price_window:d-1]
        med = past_price.median(axis=1).values
        feat["sell_price"] = cur_price
        feat["price_vs_median"] = (cur_price / med).astype("float32")

        # ---- 4. calendar / event / snap from known ----
        for c in ["dayofweek", "is_weekend", "month", "dayofyear",
                  "event_name_1", "event_type_1", "snap",
                  "item_id", "dept_id", "store_id", "state_id"]:
            feat[c] = known[c].values

        # restore categorical dtype (LightGBM needs it)
        for c in categorical_feats:
            feat[c] = feat[c].astype("category")

        # ---- 5. predict, clip negatives, write back ----
        X = feat[feature_cols]
        yhat = model.predict(X, num_iteration=model.best_iteration)
        yhat = np.clip(yhat, 0, None)
        sales_wide[d] = yhat            # write prediction back for future lags

        preds_all.append(pd.DataFrame({"id": series_ids, "d_num": d, "prediction": yhat}))

    return pd.concat(preds_all, ignore_index=True)