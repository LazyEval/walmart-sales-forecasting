from walmart_model.data_models.item import Item


class ItemSplitter:
    def __init__(self, max_forecast_horizon):
        self.max_forecast_horizon = max_forecast_horizon

    def split_items(self, items, forecast_horizon):
        train_items = []
        valid_items = []
        for item in items:
            last_record_idx = len(item.records)
            train_items.append(Item(id=item.id, records=item.records[:-forecast_horizon]))
            valid_items.append(
                Item(
                    id=item.id,
                    records=item.records[
                        last_record_idx
                        - (self.max_forecast_horizon - forecast_horizon + 7) : last_record_idx
                        - (self.max_forecast_horizon - forecast_horizon)
                    ],
                )
            )
        return train_items, valid_items
