from datetime import timedelta

from walmart_model.data_models.prediction import Forecast, Prediction


class WalmartModel:
    def __init__(self, preprocessor, predictor):
        self.preprocessor = preprocessor
        self.predictor = predictor

    def predict(self, items, forecast_period):
        inputs = self.preprocessor.preprocess_inputs(items=items, train=False)
        predictions = []
        for i, item in enumerate(items):
            predictions.append(
                Prediction(
                    item_id=item.id,
                    forecasts=[
                        Forecast(date=record.date + timedelta(forecast_period), sales=prediction)
                        for record, prediction in zip(
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
