import polars as pl


class ItemFeatureService:
    def __init__(self, lags, rolling_params):
        self.lags = lags
        self.rolling_params = rolling_params

    def add_lag_sales_features(self, frame, forecast_horizon, lags):
        return frame.with_columns(
            pl.col("sales")
            .shift(forecast_horizon + lag, fill_value=float("nan"))
            .over("id")
            .alias(f"sales_lag_{lag}")
            for lag in lags
        )

    def add_rolling_sales_features(self, frame, forecast_horizon, window_size, seasonal=False):
        window_cols = ["id"]
        seasonal_prefix = ""
        if seasonal:
            window_cols += ["weekday"]
            seasonal_prefix += "seasonal_"

        return frame.with_columns(
            pl.col("sales")
            .shift(forecast_horizon, fill_value=float("nan"))
            .rolling_mean(window_size=f"{window_size}i", min_periods=1, closed="right")
            .over(window_cols)
            .alias(f"sales_{seasonal_prefix}rolling_mean_{window_size}"),
            pl.col("sales")
            .shift(forecast_horizon, fill_value=float("nan"))
            .rolling_std(window_size=f"{window_size}i", min_periods=1, closed="right")
            .over(window_cols)
            .alias(f"sales_{seasonal_prefix}rolling_std_{window_size}"),
        )

    def add_features(self, frame, forecast_horizon):
        frame = frame.with_columns(pl.col("date").dt.weekday().alias("weekday")).pipe(
            self.add_lag_sales_features, forecast_horizon, lags=self.lags
        )
        for params in self.rolling_params:
            frame = frame.pipe(self.add_rolling_sales_features, forecast_horizon, **params)
        return frame.drop("weekday")
