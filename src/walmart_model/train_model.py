import time

import lightgbm as lgb
import typer

from config import logger
from walmart_model.components.item_loader import ItemLoader
from walmart_model.components.item_splitter import ItemSplitter
from walmart_model.components.preprocessor import Preprocessor
from walmart_model.components.scorer import Scorer
from walmart_model.components.training_pipeline import TrainingPipeline
from walmart_model.components.walmart_model import WalmartModel

TRANSACTIONS_FILE_NAME = "transactions_data_sampled.csv"
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
        item_loader=ItemLoader(
            file_name=TRANSACTIONS_FILE_NAME,
            forecast_horizon=FORECAST_HORIZON,
        ),
        item_splitter=ItemSplitter(forecast_horizon=FORECAST_HORIZON),
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
