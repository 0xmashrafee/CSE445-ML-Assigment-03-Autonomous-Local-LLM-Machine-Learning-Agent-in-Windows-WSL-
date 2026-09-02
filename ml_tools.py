# ml_tools.py
# CSE445 Assignment 3 - Mashrafe Bin Morshed
#
# All the ML "tools" the agent is allowed to call. Every function takes plain
# keyword arguments and returns a JSON string. Nothing raises an exception up
# to the caller - if something goes wrong we catch it and return
# {"error": "..."} so the agent's self-healing logic has something to react to.

import json
import time

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

import torch
import torch.nn as nn

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def _get_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS.keys())}.")
    data = DATASETS[name]()
    return data.data, data.target


def _err(msg):
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Task 1: baseline tool - dataset summary
# ---------------------------------------------------------------------------

def load_dataset_summary(dataset_name):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    classes, counts = np.unique(y, return_counts=True)
    return json.dumps({
        "dataset_name": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
    })


# ---------------------------------------------------------------------------
# Task 1: baseline tool - classic sklearn models
# ---------------------------------------------------------------------------

def train_sklearn_model(dataset_name, model_type, test_size=0.2, random_state=42):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    if not (0.05 <= test_size <= 0.9):
        return _err(f"test_size={test_size} is out of a sane range (0.05-0.9).")

    if model_type == "decision_tree":
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
    else:
        return _err(f"Unknown model_type '{model_type}'. Choose decision_tree, logistic_regression, or random_forest.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)
    test_acc = model.score(X_test_s, y_test)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)

    return json.dumps({
        "dataset_name": dataset_name,
        "model_type": model_type,
        "test_accuracy": round(float(test_acc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
    })


# ---------------------------------------------------------------------------
# Task 1: baseline tool - simple PyTorch MLP
# ---------------------------------------------------------------------------

def train_pytorch_mlp(dataset_name, hidden_size=16, epochs=20, lr=0.01, optimizer_name="adam"):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    if lr <= 0 or lr > 100:
        return _err(f"lr={lr} is not a usable learning rate.")
    if epochs <= 0:
        return _err(f"epochs={epochs} must be positive.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y))

    model = nn.Sequential(
        nn.Linear(n_features, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_classes),
    )

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return _err(f"Unknown optimizer_name '{optimizer_name}'. Use adam or sgd.")

    loss_fn = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        if torch.isnan(loss):
            return _err(f"Training diverged to NaN loss at epoch {epoch} (lr={lr}, optimizer={optimizer_name}). Try a smaller lr.")
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_t).argmax(dim=1).numpy()
    test_acc = float((preds == y_test).mean())

    return json.dumps({
        "dataset_name": dataset_name,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer_name": optimizer_name,
        "final_train_loss": round(float(loss.item()), 4),
        "test_accuracy": round(test_acc, 4),
    })


# ---------------------------------------------------------------------------
# Task 2: advanced tool - hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(dataset_name, model_type, search_type="grid", cv=5):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    X = StandardScaler().fit_transform(X)

    if model_type == "svc":
        base = SVC()
        param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    elif model_type == "decision_tree":
        base = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": [2, 4, 6, None], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]}
    else:
        return _err(f"Unknown model_type '{model_type}' for tuning. Use svc or decision_tree.")

    if search_type == "grid":
        search = GridSearchCV(base, param_grid, cv=cv, n_jobs=-1)
    elif search_type == "random":
        search = RandomizedSearchCV(base, param_grid, cv=cv, n_iter=8, random_state=42, n_jobs=-1)
    else:
        return _err(f"Unknown search_type '{search_type}'. Use grid or random.")

    search.fit(X, y)

    return json.dumps({
        "dataset_name": dataset_name,
        "model_type": model_type,
        "search_type": search_type,
        "best_params": search.best_params_,
        "best_cv_accuracy": round(float(search.best_score_), 4),
    })


# ---------------------------------------------------------------------------
# Task 2: advanced tool - dimensionality reduction / feature selection
# ---------------------------------------------------------------------------

def reduce_dimensionality(dataset_name, method="pca", n_components=2):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    max_components = min(X.shape[0], X.shape[1])
    if n_components < 1 or n_components > max_components:
        return _err(f"n_components={n_components} is invalid for this dataset (must be 1-{max_components}).")

    X = StandardScaler().fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)
        extra = {"explained_variance_ratio": [round(float(v), 4) for v in reducer.explained_variance_ratio_]}
    elif method == "sfs":
        base = LogisticRegression(max_iter=1000)
        reducer = SequentialFeatureSelector(base, n_features_to_select=n_components)
        X_reduced = reducer.fit_transform(X, y)
        extra = {"selected_feature_indices": [int(i) for i in np.where(reducer.get_support())[0]]}
    else:
        return _err(f"Unknown method '{method}'. Use pca or sfs.")

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X_reduced, y, cv=5)

    result = {
        "dataset_name": dataset_name,
        "method": method,
        "n_components": n_components,
        "reduced_accuracy": round(float(scores.mean()), 4),
    }
    result.update(extra)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Task 2: advanced tool - deeper PyTorch classifier with regularization
# ---------------------------------------------------------------------------

class DeepMLP(nn.Module):
    def __init__(self, n_features, hidden_sizes, n_classes, dropout=0.3):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_deep_pytorch_classifier(dataset_name, hidden_sizes=(32, 16), dropout=0.3, epochs=30,
                                   lr=0.001, optimizer_name="adam", batch_size=16, use_scheduler=True):
    try:
        X, y = _get_dataset(dataset_name)
    except ValueError as e:
        return _err(str(e))

    if batch_size < 2:
        return _err(f"batch_size={batch_size} is too small - BatchNorm1d needs at least 2 samples per batch.")
    if lr <= 0 or lr > 100:
        return _err(f"lr={lr} is not a usable learning rate.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if len(X_train) < batch_size:
        return _err(f"batch_size={batch_size} is larger than the training set ({len(X_train)} samples).")

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y))
    model = DeepMLP(n_features, list(hidden_sizes), n_classes, dropout=dropout)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        return _err(f"Unknown optimizer_name '{optimizer_name}'. Use adam or sgd.")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if use_scheduler else None
    loss_fn = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    if len(loader) == 0:
        return _err(f"batch_size={batch_size} with drop_last leaves zero batches for {len(X_train)} training samples.")

    final_loss = None
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            if torch.isnan(loss):
                return _err(f"Training diverged to NaN loss at epoch {epoch} (lr={lr}, optimizer={optimizer_name}). Try a smaller lr or add gradient clipping.")
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(torch.tensor(X_train, dtype=torch.float32)).argmax(dim=1).numpy()
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    train_acc = float((train_preds == y_train).mean())
    test_acc = float((test_preds == y_test).mean())

    return json.dumps({
        "dataset_name": dataset_name,
        "hidden_sizes": list(hidden_sizes),
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "optimizer_name": optimizer_name,
        "batch_size": batch_size,
        "use_scheduler": use_scheduler,
        "final_train_loss": round(float(final_loss), 4),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "overfit_gap": round(train_acc - test_acc, 4),
    })


# ---------------------------------------------------------------------------
# Tool registry the agent reads from
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_dimensionality": reduce_dimensionality,
    "train_deep_pytorch_classifier": train_deep_pytorch_classifier,
}

TOOL_DESCRIPTIONS = {
    "load_dataset_summary": (
        "Get basic info about a dataset. Args: dataset_name (iris|wine|breast_cancer)."
    ),
    "train_sklearn_model": (
        "Train a classic sklearn classifier and report test + 5-fold CV accuracy. "
        "Args: dataset_name (iris|wine|breast_cancer), model_type (decision_tree|logistic_regression|random_forest), "
        "test_size (float, optional), random_state (int, optional)."
    ),
    "train_pytorch_mlp": (
        "Train a small 1-hidden-layer PyTorch neural net. "
        "Args: dataset_name, hidden_size (int, optional), epochs (int, optional), lr (float, optional), "
        "optimizer_name (adam|sgd, optional)."
    ),
    "tune_hyperparameters": (
        "Run GridSearchCV or RandomizedSearchCV over a model's hyperparameters. "
        "Args: dataset_name, model_type (svc|decision_tree), search_type (grid|random, optional), cv (int, optional)."
    ),
    "reduce_dimensionality": (
        "Reduce feature count with PCA or sequential feature selection, then measure accuracy on the reduced features. "
        "Args: dataset_name, method (pca|sfs), n_components (int)."
    ),
    "train_deep_pytorch_classifier": (
        "Train a deeper PyTorch MLP with BatchNorm, Dropout, and an optional LR scheduler. "
        "Args: dataset_name, hidden_sizes (list of int, optional), dropout (float, optional), epochs (int, optional), "
        "lr (float, optional), optimizer_name (adam|sgd, optional), batch_size (int, optional), use_scheduler (bool, optional)."
    ),
}
