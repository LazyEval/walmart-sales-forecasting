from datetime import date

from pydantic import BaseModel


class Forecast(BaseModel):
    date: date
    sales: float


class Prediction(BaseModel):
    item_id: str
    forecasts: list[Forecast]
