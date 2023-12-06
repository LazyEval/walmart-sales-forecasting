from datetime import datetime

from pydantic import BaseModel


class Transaction(BaseModel):
    datetime: datetime
    id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
