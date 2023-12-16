from datetime import date
from typing import Union

from pydantic import BaseModel


class SalesRecord(BaseModel):
    date: date
    sales: Union[int, float]


class Item(BaseModel):
    id: str
    sales_records: list[SalesRecord]
