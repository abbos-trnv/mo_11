import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos, random_state=42):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def __init__(
        self,
        n_features,
        new_dim,
        func: Callable = np.cos,
        random_state=42,
        max_sigma_pairs=1_000_000,
        batch_size=8192,
    ):
        super().__init__(n_features, new_dim, func=func, random_state=random_state)
        self.max_sigma_pairs = max_sigma_pairs
        self.batch_size = batch_size
        self.sigma_squared = None

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _estimate_sigma_squared(self, X, rng):
        n_objects = X.shape[0]
        if n_objects < 2:
            return 1.0

        n_pairs = min(self.max_sigma_pairs, n_objects * (n_objects - 1) // 2)
        distances = np.empty(n_pairs, dtype=np.float64)
        filled = 0

        while filled < n_pairs:
            cur = min(self.batch_size, n_pairs - filled)
            i = rng.integers(0, n_objects, size=cur)
            j = rng.integers(0, n_objects - 1, size=cur)
            j = j + (j >= i)
            diff = X[i] - X[j]
            distances[filled:filled + cur] = np.einsum("ij,ij->i", diff, diff)
            filled += cur

        sigma_squared = float(np.median(distances))
        return sigma_squared if sigma_squared > 0 else 1.0

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        rng = self._rng()
        self.new_dim = X.shape[1]
        self.sigma_squared = self._estimate_sigma_squared(X, rng)
        sigma = np.sqrt(self.sigma_squared)
        self.w = rng.normal(loc=0.0, scale=1.0 / sigma, size=(self.new_dim, self.n_features))
        self.b = rng.uniform(-np.pi, np.pi, size=self.n_features)
        return self

    def transform(self, X, y=None):
        if self.w is None or self.b is None:
            raise RuntimeError("RandomFeatureCreator must be fitted before transform.")
        X = np.asarray(X, dtype=np.float64)
        return np.sqrt(2.0 / self.n_features) * self.func(X @ self.w + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        rng = self._rng()
        self.new_dim = X.shape[1]
        self.sigma_squared = self._estimate_sigma_squared(X, rng)
        sigma = np.sqrt(self.sigma_squared)

        blocks = []
        while sum(block.shape[0] for block in blocks) < self.n_features:
            gaussian = rng.normal(size=(self.new_dim, self.new_dim))
            q, _ = np.linalg.qr(gaussian)
            radii = np.sqrt(rng.chisquare(df=self.new_dim, size=self.new_dim))
            blocks.append(np.diag(radii) @ q)

        w_rows = np.vstack(blocks)[:self.n_features]
        self.w = (w_rows / sigma).T
        self.b = rng.uniform(-np.pi, np.pi, size=self.n_features)
        return self


class RFFPipeline(BaseEstimator):
    """
    Pipeline with optional PCA, random Fourier features, and a linear classifier.
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
            random_state=42,
            standardize_features: bool = True,
    ):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.feature_creator_class = feature_creator_class
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.func = func
        self.random_state = random_state
        self.standardize_features = standardize_features
        self.pipeline = None

    def fit(self, X, y):
        classifier_params = dict(self.classifier_params or {})
        if self.classifier_class is LogisticRegression:
            classifier_params.setdefault("C", 10.0)
            classifier_params.setdefault("max_iter", 1000)
        if "random_state" not in classifier_params:
            try:
                self.classifier_class(random_state=self.random_state)
                classifier_params["random_state"] = self.random_state
            except TypeError:
                pass

        pipeline_steps = []
        if self.use_PCA:
            pipeline_steps.append(
                ("pca", PCA(n_components=self.new_dim, random_state=self.random_state))
            )

        pipeline_steps.extend([
            (
                "features",
                self.feature_creator_class(
                    n_features=self.n_features,
                    new_dim=self.new_dim,
                    func=self.func,
                    random_state=self.random_state,
                ),
            ),
        ])
        if self.standardize_features:
            pipeline_steps.append(("scaler", StandardScaler()))
        pipeline_steps.append(("classifier", self.classifier_class(**classifier_params)))
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError("RFFPipeline must be fitted before predict_proba.")
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("RFFPipeline must be fitted before predict.")
        return self.pipeline.predict(X)


class RFFRegressor(BaseEstimator):
    """
    RFF approximation of an RBF-kernel regressor with a linear Ridge model.
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=RandomFeatureCreator,
            regressor_class=Ridge,
            regressor_params=None,
            func=np.cos,
            random_state=42,
    ):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.feature_creator_class = feature_creator_class
        self.regressor_class = regressor_class
        self.regressor_params = regressor_params
        self.func = func
        self.random_state = random_state
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps = []
        if self.use_PCA:
            n_components = min(self.new_dim, X.shape[1])
            pipeline_steps.append(
                ("pca", PCA(n_components=n_components, random_state=self.random_state))
            )

        pipeline_steps.extend([
            (
                "features",
                self.feature_creator_class(
                    n_features=self.n_features,
                    new_dim=self.new_dim,
                    func=self.func,
                    random_state=self.random_state,
                ),
            ),
            ("regressor", self.regressor_class(**dict(self.regressor_params or {}))),
        ])
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("RFFRegressor must be fitted before predict.")
        return self.pipeline.predict(X)
