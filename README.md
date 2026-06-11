# ML Homework 11: Random Fourier Features

This repository contains reusable code for HSE ML homework 11:

- `homework_practice_11_rff.py` - Random Fourier Features, Orthogonal Random Features, classification pipeline, and RFF regression.
- `homework_practice_11_kernel_regression.py` - Kernel Ridge Regression with gradient descent and closed-form solution.
- `homework-practice-11-random-features-solved.ipynb` - solved notebook.

## Use In Kaggle

Clone the repository into `/kaggle/working`:

```python
!git clone https://github.com/<your-username>/<repo-name>.git /kaggle/working/ml11
import sys
sys.path.append("/kaggle/working/ml11")
```

Then import the modules:

```python
from homework_practice_11_rff import RFFPipeline, RandomFeatureCreator
from homework_practice_11_kernel_regression import KernelRidgeRegression
```

To update files after pushing changes to GitHub:

```python
%cd /kaggle/working/ml11
!git pull
```

## Download Only Python Files

If you do not need the notebook, use raw file links:

```python
!wget -O homework_practice_11_rff.py https://raw.githubusercontent.com/<your-username>/<repo-name>/main/homework_practice_11_rff.py
!wget -O homework_practice_11_kernel_regression.py https://raw.githubusercontent.com/<your-username>/<repo-name>/main/homework_practice_11_kernel_regression.py
```

## Main Task Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from homework_practice_11_rff import RFFPipeline, RandomFeatureCreator

pipeline = RFFPipeline(
    n_features=1000,
    new_dim=50,
    use_PCA=True,
    feature_creator_class=RandomFeatureCreator,
    classifier_class=LogisticRegression,
    classifier_params={"C": 10.0, "max_iter": 1000, "n_jobs": -1, "multi_class": "auto"},
    random_state=42,
)

pipeline.fit(x_train, y_train)
accuracy_score(y_test, pipeline.predict(x_test))
```
