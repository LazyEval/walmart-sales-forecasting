from datetime import date

from pydantic import BaseModel


class SalesRecord(BaseModel):
    date: date
    sales: int


class Item(BaseModel):
    id: str
    sales_records: list[SalesRecord]
