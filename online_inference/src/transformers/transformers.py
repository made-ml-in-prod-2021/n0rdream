from sklearn.base import TransformerMixin


class CustomStandardScaler(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.to_numpy()
        x_transformed = (x - x.mean(axis=0)) / x.std(axis=0)
        return x_transformed
