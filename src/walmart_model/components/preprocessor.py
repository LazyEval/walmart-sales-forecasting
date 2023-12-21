import numpy as np
from sklearn.preprocessing import OrdinalEncoder

N_CAT_FEATURES = 5
N_NUM_FEATURES = 2


class Preprocessor:
    def __init__(self, encoder=None):
        if encoder is None:
            encoder = OrdinalEncoder()
        self.encoder = encoder

    def preprocess_inputs(self, items, train):
        n_samples = len([r.sales for i in items for r in i.records])
        cat_features = np.empty((n_samples, N_CAT_FEATURES), dtype=str)
        num_features = np.empty((n_samples, N_NUM_FEATURES))
        i = 0
        for item in items:
            item_tokens = item.id.split("_")
            for record in item.records:
                cat_features[i, 0] = item_tokens[0] + "_" + item_tokens[1] + "_" + item_tokens[2]
                cat_features[i, 1] = item_tokens[0] + "_" + item_tokens[1]
                cat_features[i, 2] = item_tokens[0]
                cat_features[i, 3] = item_tokens[3] + "_" + item_tokens[4]
                cat_features[i, 4] = item_tokens[3]
                num_features[i, 0] = record.sales_lag_1
                num_features[i, 1] = record.sales_lag_7
                i += 1
        if train:
            return np.hstack([self.encoder.fit_transform(cat_features), num_features])
        return np.hstack([self.encoder.transform(cat_features), num_features])
