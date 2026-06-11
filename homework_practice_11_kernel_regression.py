import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression for the dual objective

        1/2 ||K w - y||^2 + regularization / 2 * w^T K w.
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
    ):
        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None
        self.x_train: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel_scale = kernel_scale
        self.kernel = RBF(kernel_scale)

    def _kernel_matrix(self, x_left: np.ndarray, x_right: np.ndarray) -> np.ndarray:
        return self.kernel(np.asarray(x_left), np.asarray(x_right))

    def _require_fitted(self):
        if self.w is None or self.x_train is None:
            raise RuntimeError("KernelRidgeRegression must be fitted before prediction.")

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset.
        If x is the training data, this is the optimized empirical objective.
        """
        self._require_fitted()
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        k = self._kernel_matrix(x, self.x_train)
        prediction = k @ self.w
        train_k = self._kernel_matrix(self.x_train, self.x_train)
        return float(
            0.5 * np.sum((prediction - y) ** 2)
            + 0.5 * self.regularization * self.w.T @ train_k @ self.w
        )

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculating full gradient for the training dataset.
        Gradient: K (K w - y) + regularization * K w.
        """
        self._require_fitted()
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        k = self._kernel_matrix(x, x)
        return k @ (k @ self.w - y) + self.regularization * k @ self.w

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Fitting parameters with gradient descent.
        """
        self.x_train = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n_objects = self.x_train.shape[0]
        k = self._kernel_matrix(self.x_train, self.x_train)
        self.w = np.zeros(n_objects, dtype=np.float64)
        self.loss_history = []

        for _ in range(self.max_iter):
            old_w = self.w.copy()
            grad = k @ (k @ self.w - y) + self.regularization * k @ self.w
            self.w -= self.lr * grad
            self.loss_history.append(
                float(
                    0.5 * np.sum((k @ self.w - y) ** 2)
                    + 0.5 * self.regularization * self.w.T @ k @ self.w
                )
            )
            if np.sum((self.w - old_w) ** 2) < self.tolerance:
                break
        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Fitting parameters with the analytical solution:
        (K + regularization * I) w = y.
        """
        self.x_train = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        k = self._kernel_matrix(self.x_train, self.x_train)
        regularized_k = k + self.regularization * np.eye(k.shape[0])
        self.w = np.linalg.solve(regularized_k, y)
        self.loss_history = [self.calc_loss(self.x_train, y)]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset.
        """
        self._require_fitted()
        k = self._kernel_matrix(np.asarray(x, dtype=np.float64), self.x_train)
        return k @ self.w
