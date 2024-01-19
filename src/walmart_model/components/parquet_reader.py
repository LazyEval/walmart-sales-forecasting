import polars as pl

from config import config


class ParquetReader:
    def __init__(self, file_name):
        self.file_path = config.RAW_DATA_DIR / file_name

    def get_data(self):
        return pl.scan_parquet(self.file_path).with_columns(pl.col("date").dt.date())
