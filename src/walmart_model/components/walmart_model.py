from datetime import timedelta

from walmart_model.data_models.prediction import Forecast, Prediction


class WalmartModel:
    def __init__(self, preprocessor, predictor, forecast_horizon):
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.forecast_horizon = forecast_horizon

    def train(self, items):
        targets = [r.sales for i in items for r in i.records]
        inputs = self.preprocessor.preprocess_inputs(items=items, train=True)
        self.predictor.fit(inputs, targets)

    def predict(self, items):
        inputs = self.preprocessor.preprocess_inputs(items=items, train=False)
        predictions = []
        for i, item in enumerate(items):
            predictions.append(
                Prediction(
                    item_id=item.id,
                    forecasts=[
                        Forecast(date=j.date + timedelta(self.forecast_horizon), sales=k)
                        for j, k in zip(
                            item.records,
                            self.predictor.predict(inputs)[
                                i * self.forecast_horizon : (i + 1) * self.forecast_horizon
                            ],
                        )
                    ],
                )
            )
        return predictions

    def get_params(self):
        return self.predictor.get_params()
