import itertools
from statistics import mean


class Scorer:
    def __init__(self):
        pass

    def get_rmsse(self, train_items, valid_items, forecasts):
        scale_per_item = {}
        for item in train_items:
            scale_per_item[item.id] = mean(
                [
                    (i - j) ** 2
                    for i, j in itertools.pairwise([i.sales for i in item.sales_records])
                ]
            )
        mse_per_item = {}
        for item, forecast in zip(valid_items, forecasts):
            mse_per_item[item.id] = mean(
                [
                    (i - j) ** 2
                    for i, j in zip(
                        [i.sales for i in item.sales_records],
                        [i.sales for i in forecast.sales_records],
                    )
                ]
            )
        return mean([(mse_per_item[k] / scale_per_item[k]) ** 0.5 for k in list(scale_per_item)])
