class ItemSplitter:
    def __init__(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon

    def split_items(self, items_frame):
        valid_items = items_frame.group_by("id").tail(self.forecast_horizon)
        train_items = items_frame.join(valid_items, on=["date", "id"], how="anti")
        return train_items, valid_items
