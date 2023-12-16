from walmart_model.data_models.item import Item, SalesRecord


class WalmartModel:
    def __init__(self, preprocessor, predictor, forecast_horizon):
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.forecast_horizon = forecast_horizon

    def train(self, items):
        targets = [i.sales for j in items for i in j.sales_records]
        inputs = self.preprocessor.preprocess_inputs(items, train=True)
        self.predictor.fit(inputs, targets)

    def forecast(self, items):
        inputs = self.preprocessor.preprocess_inputs(items, train=False)
        forecasts = []
        for i, item in enumerate(items):
            forecasts.append(
                Item(
                    id=item.id,
                    sales_records=[
                        SalesRecord(
                            date=j.date,
                            sales=k,
                        )
                        for j, k in zip(
                            item.sales_records,
                            self.predictor.predict(inputs)[
                                i * self.forecast_horizon : (i + 1) * self.forecast_horizon
                            ],
                        )
                    ],
                )
            )
        return forecasts

    def get_params(self):
        return self.predictor.get_params()
