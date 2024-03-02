from config import logger


class TrainingPipeline:
    def __init__(self, item_loader, item_splitter, model_trainer, scorer):
        self.item_loader = item_loader
        self.item_splitter = item_splitter
        self.model_trainer = model_trainer
        self.scorer = scorer

    def run(self):
        items_per_forecast_horizon = self.item_loader.get_items_per_forecast_horizon()
        valid_items = []
        predictions = []
        for forecast_horizon, items in items_per_forecast_horizon.items():
            train_items, valid_items_batch = self.item_splitter.split_items(
                items, forecast_horizon
            )
            model = self.model_trainer.get_model(train_items, valid_items_batch)
            valid_items += valid_items_batch
            predictions += model.predict(valid_items_batch)
        rmsse = self.scorer.get_rmsse(train_items, valid_items, predictions)
        logger.logger.info(f"Validation RMSSE: {rmsse:.3f}\n{len(train_items)} items")
