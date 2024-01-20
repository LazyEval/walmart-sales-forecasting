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
    sales_lag_28: Optional[float] = None
    sales_rolling_mean_7: Optional[float] = None
    sales_rolling_mean_28: Optional[float] = None
    sales_rolling_std_7: Optional[float] = None
    sales_rolling_std_28: Optional[float] = None
    sales_seasonal_rolling_mean_4: Optional[float] = None
    sales_seasonal_rolling_mean_8: Optional[float] = None
    sales_seasonal_rolling_std_4: Optional[float] = None
    sales_seasonal_rolling_std_8: Optional[float] = None


class Item(BaseModel):
    id: str
    records: list[Record]
