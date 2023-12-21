from walmart_model.data_models.item import Item


class ItemSplitter:
    def __init__(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon

    def split_items(self, items):
        train_items = []
        valid_items = []
        for item in items:
            train_items.append(Item(id=item.id, records=item.records[: -self.forecast_horizon]))
            valid_items.append(Item(id=item.id, records=item.records[-self.forecast_horizon :]))
        return train_items, valid_items
