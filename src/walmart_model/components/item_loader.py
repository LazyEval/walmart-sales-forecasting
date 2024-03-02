import polars as pl

from walmart_model.data_models.item import Item, Record


class ItemLoader:
    def __init__(self, item_reader, price_reader, calendar_reader, forecast_horizons):
        self.item_reader = item_reader
        self.price_reader = price_reader
        self.calendar_reader = calendar_reader
        self.forecast_horizons = forecast_horizons

    def add_lag_sales_features(self, frame, forecast_horizon, lags):
        return frame.with_columns(
            pl.col("sales").shift(forecast_horizon + lag).over("id").alias(f"sales_lag_{lag}")
            for lag in lags
        )

    def add_rolling_sales_features(self, frame, forecast_horizon, window_size):
        return frame.with_columns(
            pl.col("sales")
            .shift(forecast_horizon)
            .rolling_mean(window_size=f"{window_size}i", min_periods=1, closed="right")
            .over("id")
            .alias(f"sales_rolling_mean_{window_size}"),
            pl.col("sales")
            .shift(forecast_horizon)
            .rolling_std(window_size=f"{window_size}i", min_periods=1, closed="right")
            .over("id")
            .alias(f"sales_rolling_std_{window_size}"),
        )

    def add_seasonal_rolling_sales_features(self, frame, forecast_horizon, window_size):
        return (
            frame.with_columns(pl.col("date").dt.weekday().alias("weekday"))
            .with_columns(
                pl.col("sales")
                .shift(forecast_horizon)
                .rolling_mean(window_size=f"{window_size}i", min_periods=1, closed="right")
                .over("id", "weekday")
                .alias(f"sales_seasonal_rolling_mean_{window_size}"),
                pl.col("sales")
                .shift(forecast_horizon)
                .rolling_std(window_size=f"{window_size}i", min_periods=1, closed="right")
                .over("id", "weekday")
                .alias(f"sales_seasonal_rolling_std_{window_size}"),
            )
            .drop("weekday")
        )

    def get_sorted_items(self):
        return (
            self.item_reader.get_raw_items()
            .join(self.price_reader.get_data().collect(), on=["date", "item_id", "store_id"])
            .join(
                self.calendar_reader.get_data()
                .select(pl.col("date"), pl.col("snap_TX").alias("holiday"))
                .collect(),
                on="date",
            )
            .sort(["id", "date"])
        )

    def add_features(self, frame, forecast_horizon):
        return (
            frame.pipe(self.add_lag_sales_features, forecast_horizon, lags=[1, 7, 28])
            .pipe(self.add_rolling_sales_features, forecast_horizon, window_size=7)
            .pipe(self.add_rolling_sales_features, forecast_horizon, window_size=28)
            .pipe(self.add_seasonal_rolling_sales_features, forecast_horizon, window_size=4)
            .pipe(self.add_seasonal_rolling_sales_features, forecast_horizon, window_size=8)
        )

    def get_items_per_forecast_horizon(self):
        return {
            forecast_horizon: [
                Item(id=k, records=[Record(**i) for i in v])
                for k, v in self.add_features(self.get_sorted_items(), forecast_horizon)
                .rows_by_key(key=["id"], named=True)
                .items()
            ]
            for forecast_horizon in self.forecast_horizons
        }
