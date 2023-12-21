import polars as pl

from config import config
from walmart_model.data_models.item import Item, Record

LAGS = [1, 7]


class ItemLoader:
    def __init__(self, file_name, forecast_horizon):
        self.file_path = config.RAW_DATA_DIR / file_name
        self.forecast_horizon = forecast_horizon

    def add_lag_features(self, frame, col_name):
        return frame.with_columns(
            pl.col(col_name)
            .shift(self.forecast_horizon + lag)
            .over("id")
            .alias(f"{col_name}_lag_{lag}")
            for lag in LAGS
        )

    def get_items_frame(self):
        item_sales = (
            pl.read_csv(self.file_path, try_parse_dates=True)
            .with_columns(pl.col("date").dt.date())
            .group_by("date", "id")
            .agg(pl.col("id").count().alias("sales"))
        )
        return (
            pl.DataFrame(
                {
                    "date": pl.date_range(
                        item_sales.select(pl.col("date")).min().item(),
                        item_sales.select(pl.col("date")).max().item(),
                        "1d",
                        eager=True,
                    )
                }
            )
            .join(item_sales.select(pl.col("id").unique()), how="cross")
            .join(item_sales, on=["date", "id"], how="outer")
            .with_columns(pl.col("sales").fill_null(0))
            .filter(pl.col("sales").cumsum().over("id") > 0)  # Drop records until first sales
            .sort(["date", "id"])
            .pipe(self.add_lag_features, col_name="sales")
            .lazy()
        )

    def get_items(self):
        return [
            Item(id=k, item_history=[Record(**i) for i in v])
            for k, v in self.get_items_frame()
            .collect()
            .rows_by_key(key=["id"], named=True)
            .items()
        ]
