import polars as pl

from walmart_model.data_models.item import Item, Record


class ItemLoader:
    def __init__(
        self, item_reader, price_reader, calendar_reader, item_feature_service, forecast_horizons
    ):
        self.item_reader = item_reader
        self.price_reader = price_reader
        self.calendar_reader = calendar_reader
        self.item_feature_service = item_feature_service
        self.forecast_horizons = forecast_horizons

    def get_items(self):
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

    def get_items_per_forecast_horizon(self):
        items = self.get_items()
        return {
            forecast_horizon: [
                Item(id=item_id, records=[Record(**record) for record in records])
                for item_id, records in self.item_feature_service.add_features(
                    items, forecast_horizon
                )
                .rows_by_key(key=["id"], named=True)
                .items()
            ]
            for forecast_horizon in self.forecast_horizons
        }
