from config import logger


class TrainingPipeline:
    def __init__(self, train_loader, valid_loader, model, scorer):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.scorer = scorer

    def run(self):
        train_items = self.train_loader.get_items()
        valid_items = self.valid_loader.get_items()
        self.model.train(train_items)
        forecasts = self.model.forecast(valid_items)
        rmsse = self.scorer.get_rmsse(train_items, valid_items, forecasts)
        logger.logger.info(
            f"Validation RMSSE: {rmsse:.3f}\n"
            f"{len(train_items)} training samples\n"
            f"{len(valid_items)} validation samples"
        )
