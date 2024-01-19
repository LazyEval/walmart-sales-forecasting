import polars as pl

from walmart_model.data_models.item import Item, Record

LAGS = [1, 7]


class ItemLoader:
    def __init__(self, item_reader, price_reader, calendar_reader, forecast_horizon):
        self.item_reader = item_reader
        self.price_reader = price_reader
        self.calendar_reader = calendar_reader
        self.forecast_horizon = forecast_horizon

    def add_lag_features(self, frame, col_name):
        return frame.with_columns(
            pl.col(col_name)
            .shift(self.forecast_horizon + lag)
            .over("id")
            .alias(f"{col_name}_lag_{lag}")
            for lag in LAGS
        )

    def get_items(self):
        return [
            Item(id=k, records=[Record(**i) for i in v])
            for k, v in self.item_reader.get_raw_items()
            .join(self.price_reader.get_data().collect(), on=["date", "item_id", "store_id"])
            .join(
                self.calendar_reader.get_data()
                .select(pl.col("date"), pl.col("snap_TX").alias("holiday"))
                .collect(),
                on="date",
            )
            .pipe(self.add_lag_features, col_name="sales")
            .rows_by_key(key=["id"], named=True)
            .items()
        ]
