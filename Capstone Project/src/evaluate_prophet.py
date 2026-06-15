import pandas as pd
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)   # silence per-fit logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

def evaluate_prophet_batch(df, test_long, subset_ids, train_end, horizon=28):
    """Fit Prophet per series, forecast the test horizon, compute RMSSE/MAE.
    Negative predictions are clipped to 0 (sales can't be negative)."""
    results = []
    for i, sid in enumerate(subset_ids):
        # training data for this series
        s = df[df["id"] == sid].sort_values("date")
        train_s = s[s["d_num"] <= train_end]
        if len(train_s) < 60:           # too short to fit seasonality
            continue

        pdf = train_s[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
        try:
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        daily_seasonality=False)
            m.fit(pdf)
            future = m.make_future_dataframe(periods=horizon)
            fc = m.predict(future)
            yhat = fc["yhat"].values[-horizon:].clip(min=0)   # clip negatives
        except Exception:
            continue

        # test actuals for this series, ordered by day
        ta = (test_long[test_long["id"] == sid]
              .sort_values("d_num")["sales"].values)
        if len(ta) != horizon:
            continue

        from evaluation_index import rmsse, mae
        r = rmsse(ta, yhat, train_s["sales"].values)
        m_ = mae(ta, yhat)
        results.append({"id": sid, "rmsse": r, "mae": m_})

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(subset_ids)} done")

    return pd.DataFrame(results)