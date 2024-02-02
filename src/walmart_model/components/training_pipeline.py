from config import logger


class TrainingPipeline:
    def __init__(self, item_loader, item_splitter, model_trainer, forecast_horizon, scorer):
        self.item_loader = item_loader
        self.item_splitter = item_splitter
        self.model_trainer = model_trainer
        self.forecast_horizon = forecast_horizon
        self.scorer = scorer

    def run(self):
        items = self.item_loader.get_items()
        train_items, valid_items = self.item_splitter.split_items(items)
        model = self.model_trainer.get_model(train_items, valid_items, self.forecast_horizon)
        predictions = model.predict(valid_items)
        rmsse = self.scorer.get_rmsse(train_items, valid_items, predictions)
        logger.logger.info(f"Validation RMSSE: {rmsse:.3f}\n{len(train_items)} items")
