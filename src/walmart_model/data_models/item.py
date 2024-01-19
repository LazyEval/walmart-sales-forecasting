from datetime import date
from typing import Optional

from pydantic import BaseModel


class Record(BaseModel):
    date: date
    holiday: int
    sell_price: float
    sales: float
    sales_lag_1: Optional[float] = None
    sales_lag_7: Optional[float] = None


class Item(BaseModel):
    id: str
    records: list[Record]
