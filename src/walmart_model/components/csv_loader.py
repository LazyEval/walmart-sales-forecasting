import csv
import gzip
from datetime import datetime

from config import config
from walmart_model.data_models.transaction import Transaction


class CSVLoader:
    def __init__(self, file_name):
        self.file_path = config.RAW_DATA_DIR / f"{file_name}.csv.gz"

    def get_transactions(self):
        with gzip.open(self.file_path, "rt", encoding="utf-8-sig") as f:
            return [
                Transaction(
                    datetime=datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S"),
                    id=row["id"],
                    item_id=row["item_id"],
                    dept_id=row["dept_id"],
                    cat_id=row["cat_id"],
                    store_id=row["store_id"],
                    state_id=row["state_id"],
                )
                for row in csv.DictReader(f, delimiter=",")
            ]
