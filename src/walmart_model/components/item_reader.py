import polars as pl

from config import config


class ItemReader:
    def __init__(self, file_name):
        self.file_path = config.RAW_DATA_DIR / file_name

    def get_raw_items(self):
        item_sales = pl.scan_csv(self.file_path, try_parse_dates=True).with_columns(
            pl.col("date").dt.date()
        )
        min_date = item_sales.select(pl.col("date")).collect().min().item()
        max_date = item_sales.select(pl.col("date")).collect().max().item()
        df = (
            pl.LazyFrame({"date": pl.date_range(min_date, max_date, "1d", eager=True)})
            .join(item_sales.select(pl.col("id").unique()), how="cross")
            .join(
                item_sales.group_by("date", "id").agg(pl.col("id").count().alias("sales")),
                on=["date", "id"],
                how="outer",
            )
            .with_columns(
                [pl.col("id").str.split("_").list.get(i).alias(f"id_{i}") for i in range(5)],
            )
            .with_columns(pl.col("sales").fill_null(0))
            .filter(pl.col("sales").cumsum().over("id") > 0)  # Drop records until first sales
        )
        return df.collect().select(
            pl.col("date"),
            pl.col("id"),
            (pl.col("id_0") + "_" + pl.col("id_1") + "_" + pl.col("id_2")).alias("item_id"),
            (pl.col("id_3") + "_" + pl.col("id_4")).alias("store_id"),
            pl.col("sales"),
        )
