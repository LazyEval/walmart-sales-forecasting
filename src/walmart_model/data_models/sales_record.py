from datetime import date

from pydantic import BaseModel


class SalesRecord(BaseModel):
    date: date
    id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    sales: int
