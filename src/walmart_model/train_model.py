import time

import lightgbm as lgb
import typer

from config import logger
from walmart_model.components.item_loader import ItemLoader
from walmart_model.components.preprocessor import Preprocessor
from walmart_model.components.scorer import Scorer
from walmart_model.components.training_pipeline import TrainingPipeline
from walmart_model.components.walmart_model import WalmartModel

TRAIN_ITEMS_FILE_NAME = "train_items.json"
VALID_ITEMS_FILE_NAME = "valid_items.json"
FORECAST_HORIZON = 28
PARAMS = {
    "verbose": -1,
    "num_leaves": 256,
    "n_estimators": 50,
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
}


def main():
    start = time.time()

    pipeline = TrainingPipeline(
        train_loader=ItemLoader(file_name=TRAIN_ITEMS_FILE_NAME),
        valid_loader=ItemLoader(file_name=VALID_ITEMS_FILE_NAME),
        model=WalmartModel(
            preprocessor=Preprocessor(),
            predictor=lgb.LGBMRegressor(**PARAMS),
            forecast_horizon=FORECAST_HORIZON,
        ),
        scorer=Scorer(),
    )
    pipeline.run()

    end = time.time()
    logger.logger.info(f"Runtime: {end - start:.3} seconds")


if __name__ == "__main__":
    typer.run(main)
