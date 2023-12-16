import json

from config import config
from walmart_model.data_models.item import Item, SalesRecord


class ItemLoader:
    def __init__(self, file_name):
        self.file_path = config.PROCESSED_DATA_DIR / file_name

    def get_items(self):
        items = []
        for row in open(self.file_path, "rt"):
            data = json.loads(row)
            items.append(
                Item(
                    id=data["id"],
                    sales_records=[
                        SalesRecord(date=i, sales=j) for i, j in zip(data["date"], data["sales"])
                    ],
                )
            )
        return items
