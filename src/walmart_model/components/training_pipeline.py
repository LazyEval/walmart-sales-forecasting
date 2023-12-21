from config import logger


class TrainingPipeline:
    def __init__(self, item_loader, item_splitter, model, scorer):
        self.item_loader = item_loader
        self.item_splitter = item_splitter
        self.model = model
        self.scorer = scorer

    def run(self):
        items = self.item_loader.get_items()
        train_items, valid_items = self.item_splitter.split_items(items)
        self.model.train(train_items)
        predictions = self.model.predict(valid_items)
        rmsse = self.scorer.get_rmsse(train_items, valid_items, predictions)
        logger.logger.info(f"Validaion RMSSE: {rmsse:.3f}\n{len(train_items)} items")
