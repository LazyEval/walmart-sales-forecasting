import time

import typer

from config import logger
from walmart_model.components.item_loader import ItemLoader
from walmart_model.components.item_reader import ItemReader
from walmart_model.components.item_splitter import ItemSplitter
from walmart_model.components.lightgbm_trainer import LightGBMTrainer
from walmart_model.components.parquet_reader import ParquetReader
from walmart_model.components.preprocessor import Preprocessor
from walmart_model.components.scorer import Scorer
from walmart_model.components.training_pipeline import TrainingPipeline

TRANSACTIONS_FILE_NAME = "transactions_data_sampled.csv"
PRICES_FILE_NAME = "prices.parquet"
CALENDAR_FILE_NAME = "calendar.parquet"
FORECAST_HORIZONS = [7, 14, 21, 28]
MODEL_PARAMS = {
    "objective": "tweedie",
    "metric": "None",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "min_samples_leaf": 100,
    "subsample": 0.3,
    "feature_fraction": 0.3,
    "deterministic": True,
}
NUM_BOOST_ROUNDS = 1000
STOPPING_ROUNDS = 100
LOG_PERIOD = 50


def main():
    start = time.time()

    pipeline = TrainingPipeline(
        item_loader=ItemLoader(
            item_reader=ItemReader(file_name=TRANSACTIONS_FILE_NAME),
            price_reader=ParquetReader(file_name=PRICES_FILE_NAME),
            calendar_reader=ParquetReader(file_name=CALENDAR_FILE_NAME),
            forecast_horizons=FORECAST_HORIZONS,
        ),
        item_splitter=ItemSplitter(max_forecast_horizon=FORECAST_HORIZONS[-1]),
        model_trainer=LightGBMTrainer(
            preprocessor=Preprocessor(),
            model_params=MODEL_PARAMS,
            num_boost_rounds=NUM_BOOST_ROUNDS,
            stopping_rounds=STOPPING_ROUNDS,
            log_period=LOG_PERIOD,
        ),
        scorer=Scorer(),
    )
    pipeline.run()

    end = time.time()
    logger.logger.info(f"Runtime: {end - start:.3} seconds")


if __name__ == "__main__":
    typer.run(main)
