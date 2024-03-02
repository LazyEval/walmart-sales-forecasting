from datetime import timedelta

from walmart_model.data_models.prediction import Forecast, Prediction


class WalmartModel:
    def __init__(self, preprocessor, predictor, forecast_horizon):
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.forecast_horizon = forecast_horizon

    def predict(self, items):
        inputs = self.preprocessor.preprocess_inputs(items=items, train=False)
        forecast_period = len(items[0].records)
        predictions = []
        for i, item in enumerate(items):
            predictions.append(
                Prediction(
                    item_id=item.id,
                    forecasts=[
                        Forecast(date=record.date + timedelta(self.forecast_horizon), sales=output)
                        for record, output in zip(
                            item.records,
                            self.predictor.predict(inputs)[
                                i * forecast_period : (i + 1) * forecast_period
                            ],
                        )
                    ],
                )
            )
        return predictions

    def get_params(self):
        return self.predictor.get_params()
