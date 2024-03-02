import itertools
from statistics import mean


class Scorer:
    def __init__(self):
        pass

    def get_scale(self, item):
        return mean([(i - j) ** 2 for i, j in itertools.pairwise([i.sales for i in item.records])])

    def get_mse(self, item, prediction):
        return mean(
            [
                (i - j) ** 2
                for i, j in zip(
                    [i.sales for i in item.records],
                    [i.sales for i in prediction.forecasts],
                )
            ]
        )

    def get_rmsse(self, train_items, valid_items, predictions):
        scale_per_item = {i.id: self.get_scale(i) for i in train_items}
        mse_per_item = {i.id: self.get_mse(i, p) for i, p in zip(valid_items, predictions)}
        return mean([(mse_per_item[k] / scale_per_item[k]) ** 0.5 for k in list(scale_per_item)])
