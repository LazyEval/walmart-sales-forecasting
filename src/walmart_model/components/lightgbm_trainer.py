import itertools
from statistics import mean

import lightgbm as lgb
import numpy as np

from walmart_model.components.walmart_model import WalmartModel


class LightGBMTrainer:
    def __init__(self, preprocessor, model_params, num_boost_rounds, stopping_rounds, log_period):
        self.preprocessor = preprocessor
        self.model_params = model_params
        self.num_boost_rounds = num_boost_rounds
        self.stopping_rounds = stopping_rounds
        self.log_period = log_period

    def get_scale(self, train_items, valid_items):
        valid_fold_size = len(valid_items[0].records)
        scale = []
        for item in train_items:
            scale.extend(
                [
                    mean(
                        [
                            (i - j) ** 2
                            for i, j in itertools.pairwise([i.sales for i in item.records])
                        ]
                    )
                    for i in range(valid_fold_size)
                ]
            )
        return scale

    def rmsse(self, predictions, samples):
        rmsse = np.mean(((predictions - samples.get_label()) ** 2 / samples.get_weight()) ** 0.5)
        return "RMSSE", rmsse, False

    def get_model(self, train_items, valid_items):
        train_set = lgb.Dataset(
            self.preprocessor.preprocess_inputs(items=train_items, train=True),
            [r.sales for i in train_items for r in i.records],
        )
        valid_set = lgb.Dataset(
            self.preprocessor.preprocess_inputs(items=valid_items, train=False),
            [r.sales for i in valid_items for r in i.records],
            weight=self.get_scale(train_items, valid_items),
        )
        model = lgb.train(
            params=self.model_params,
            train_set=train_set,
            num_boost_round=self.num_boost_rounds,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(self.stopping_rounds),
                lgb.log_evaluation(self.log_period),
            ],
            feval=self.rmsse,
        )
        return WalmartModel(preprocessor=self.preprocessor, predictor=model)
