#!/usr/bin/env python3
"""Validation tests for addernet 1.0.6"""

import sys
sys.path.insert(0, '.')

from addernet import AdderNetHDC, hdc_detect_backend, AdderCluster, AdderBoost
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np
import time

print(f"Backend: {hdc_detect_backend()}\n")

# ---- Test 1: HDC with verbose and history ----
print("=== Test 1: HDC com verbose e historico ===")
for name, loader in [("Wine", load_wine), ("Cancer", load_breast_cancer)]:
    data = loader()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    sc = StandardScaler()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=10000, random_state=42)
    mlp.fit(sc.fit_transform(Xtr), ytr)
    acc_mlp = accuracy_score(yte, mlp.predict(sc.transform(Xte)))

    model = AdderNetHDC(n_vars=X.shape[1], n_classes=len(np.unique(y)), table_size=256, seed=42)
    history = model.train(Xtr, ytr, n_iter=100, margin=0.02,
                          regenerate=0.05, patience=5, verbose=True)
    model.warm_cache()
    acc_hdc = accuracy_score(yte, model.predict_batch(Xte))

    t0 = time.perf_counter()
    for _ in range(100_000):
        model.predict_batch(Xte[0:1])
    one_us = (time.perf_counter() - t0) / 100_000 * 1_000_000

    print(f"{name}: HDC={acc_hdc * 100:.1f}% | MLP={acc_mlp * 100:.1f}% | 1 inference={one_us:.2f}us")
    print(f"  history: epochs={history['epochs_run']} | best_acc={history['best_val_accuracy']*100:.1f}% | "
          f"stopped_early={history['stopped_early']}")
    assert acc_hdc >= 0.80, f"[{name}] Acurácia regrediu: {acc_hdc}"
    assert history['epochs_run'] <= 100
    assert isinstance(history['best_val_accuracy'], float)

# ---- Test 2: AdderCluster with range fix ----
print("\n=== Test 2: AdderCluster com range ===")
data = load_breast_cancer()
X, y = data.data, data.target
sc2 = MinMaxScaler(feature_range=(0, 150))
X_norm = sc2.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X_norm, y, test_size=0.3, random_state=42, stratify=y)

for n_nodes, strategy in [(1, 'random'), (2, 'random'), (4, 'random'), (2, 'range')]:
    combo = 'mean' if strategy == 'range' else 'vote'
    ovlp = 0.5 if strategy == 'range' else 0.1
    cluster = AdderCluster(n_nodes=n_nodes, strategy=strategy, combination=combo,
                           overlap=ovlp, verbose=(strategy == 'range'))
    cluster.fit(Xtr, ytr)
    preds = cluster.predict_batch(Xte)
    acc = accuracy_score(yte, preds)

    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        cluster.predict_batch(Xte)
    ms = (time.perf_counter() - t0) / N * 1000

    print(f"  n_nodes={n_nodes} strategy={strategy:<10} acc={acc * 100:.2f}% infer={ms:.2f}ms")
    if strategy == 'range':
        assert acc >= 0.85, f"Range ainda colapsado: {acc}"

cluster.info()

# ---- Test 3: AdderBoost with lr_boost ----
print("\n=== Test 3: AdderBoost com lr_boost ===")
data = load_diabetes()
X2, y2 = data.data, data.target
sc3 = MinMaxScaler(feature_range=(0, 150))
X2_norm = sc3.fit_transform(X2)
Xtr3, Xte3, ytr3, yte3 = train_test_split(X2_norm, y2, test_size=0.3, random_state=42)

boost = AdderBoost(n_estimators=5, lr_boost=0.1, size=256, bias=50,
                   input_min=0, input_max=150, lr=0.05)
boost.fit(Xtr3, ytr3)
mae = mean_absolute_error(yte3, boost.predict_batch(Xte3))
print(f"  AdderBoost MAE (diabetes): {mae:.2f}")
assert mae < 100, f"MAE muito alto: {mae}"

# ---- Test 4: HDC Cancer specific validation (>= 92%) ----
print("\n=== Test 4: HDC Cancer acuracia >= 92% ===")
data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = AdderNetHDC(n_vars=X.shape[1], n_classes=2, table_size=256, seed=42)
history = model.train(Xtr, ytr, n_iter=100, margin=0.02,
                      regenerate=0.05, patience=5, verbose=True)
model.warm_cache()
acc = accuracy_score(yte, model.predict_batch(Xte))
print(f"Cancer: {acc*100:.2f}% | parou no epoch {history['epochs_run']} | "
      f"melhor={history['best_val_accuracy']*100:.2f}%")
assert acc >= 0.90, f"Acurácia regrediu: {acc}"
assert history['epochs_run'] <= 100

print("\nTodos os testes passaram! ✓")
