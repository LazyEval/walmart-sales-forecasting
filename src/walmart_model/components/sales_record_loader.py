import polars as pl

from config import config
from walmart_model.data_models.sales_record import SalesRecord


class SalesRecordLoader:
    def __init__(self, file_name):
        self.file_path = config.RAW_DATA_DIR / file_name

    def get_sales_records(self):
        transactions = (
            pl.read_csv(self.file_path, try_parse_dates=True)
            .with_columns(pl.col("date").dt.date())
            .group_by("date", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
            .agg(pl.col("id").count().alias("sales"))
        )
        sales_records = (
            pl.DataFrame(
                {
                    "date": pl.date_range(
                        transactions.select(pl.col("date")).min().item(),
                        transactions.select(pl.col("date")).max().item(),
                        "1d",
                        eager=True,
                    )
                }
            )
            .join(transactions.select(pl.col("id").unique()), how="cross")
            .join(transactions, on=["date", "id"], how="outer")
            .with_columns(
                pl.col("^[a-z]+_id$").drop_nulls().first().over("id"),
                pl.col("sales").fill_null(0),
            )
            .sort(["date", "id"])
        )
        for row in sales_records.iter_rows(named=True):
            yield SalesRecord(**row)
