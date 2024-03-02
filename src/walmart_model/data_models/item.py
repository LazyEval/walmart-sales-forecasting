from datetime import date

from pydantic import BaseModel


class Record(BaseModel):
    date: date
    holiday: int
    sell_price: float
    sales: float
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_28: float
    sales_rolling_mean_7: float
    sales_rolling_mean_28: float
    sales_rolling_std_7: float
    sales_rolling_std_28: float
    sales_seasonal_rolling_mean_4: float
    sales_seasonal_rolling_mean_8: float
    sales_seasonal_rolling_std_4: float
    sales_seasonal_rolling_std_8: float


class Item(BaseModel):
    id: str
    records: list[Record]
