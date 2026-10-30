import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfAspectModel:
    def __init__(self, C=4.0):
        self.C = C
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
        self.clfs = []

    def _features(self, texts, fit=False):
        if fit:
            return hstack([self.words.fit_transform(texts), self.chars.fit_transform(texts)]).tocsr()
        return hstack([self.words.transform(texts), self.chars.transform(texts)]).tocsr()

    def fit(self, texts, y):
        X = self._features(texts, fit=True)
        self.clfs = []
        for j in range(y.shape[1]):
            clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
            self.clfs.append(clf.fit(X, y[:, j]))
        return self

    def predict(self, texts):
        X = self._features(texts)
        return np.column_stack([clf.predict(X) for clf in self.clfs])
