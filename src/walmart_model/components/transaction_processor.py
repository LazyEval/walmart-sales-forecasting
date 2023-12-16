import polars as pl

from config import config


class TransactionProcessor:
    def __init__(self, file_name, splitter):
        self.file_path = config.RAW_DATA_DIR / file_name
        self.splitter = splitter

    def save_items(self):
        transactions = (
            pl.read_csv(self.file_path, try_parse_dates=True)
            .with_columns(pl.col("date").dt.date())
            .group_by("date", "id")
            .agg(pl.col("id").count().alias("sales"))
        )
        items = (
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
            .with_columns(pl.col("sales").fill_null(0))
            .filter(pl.col("sales").cumsum().over("id") > 0)  # Drop records until first sales
            .sort(["date", "id"])
        )
        train_items, valid_items = self.splitter.split_items(items)
        train_items.group_by("id").agg(pl.col("date"), pl.col("sales")).write_ndjson(
            config.PROCESSED_DATA_DIR / "train_items.json"
        )
        valid_items.group_by("id").agg(pl.col("date"), pl.col("sales")).write_ndjson(
            config.PROCESSED_DATA_DIR / "valid_items.json"
        )
