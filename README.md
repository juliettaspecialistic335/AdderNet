# AdderNet

[![PyPI version](https://img.shields.io/pypi/v/addernet.svg)](https://pypi.org/project/addernet/)
[![Python](https://img.shields.io/pypi/pyversions/addernet.svg)](https://pypi.org/project/addernet/)
[![License](https://img.shields.io/github/license/PedroHenriqueBatistaSilva/AdderNet.svg)](LICENSE)

Biblioteca de machine learning que **não usa multiplicação de ponto flutuante** na inferência. Zero.

> Benchmarks medidos em CPU x86-64 com backend **AVX2**, Python 3.x, v1.0.8.

---

## O que é?

AdderNet substitui multiplicações por **lookups em tabela** (LUT) e operações de soma inteiras,
tornando a inferência viável em microcontroladores sem FPU (ESP32, STM32, RPi).

A biblioteca expõe quatro componentes principais:

| Classe | Descrição |
|---|---|
| `AdderNetLayer` | Rede de uma variável — LUT + soma, zero multiplicação |
| `AdderNetHDC` | Classificador multivariável — Hyperdimensional Computing (HDC) |
| `AdderCluster` | Ensemble de `AdderNetLayer` com estratégias de combinação |
| `AdderBoost` | Gradient Boosting com `AdderNetLayer` — inferência sem multiplicação |

---

## Novidades v1.0.8

- **OnlineHD** — treino com bundling ponderado por novidade (anti-saturação)
- **AdaptHD** — retreino iterativo com correção de erro (`n_iter`)
- **Early-exit Hamming** — aborta comparação quando classes estão bem separadas
- **AdderCluster** — ensemble multi-estratégia (`random`, `range`, `feature`, `boosting`)
- **AdderBoost** — gradient boosting 100 % LUT para regressão
- **warm_cache()** — pré-computa hipervectores para inferência máxima

---

## Instalação

```bash
pip install addernet
```

Ou do código-fonte:

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
make
pip install -e .
```

---

## Uso — AdderNetLayer (uma variável)

```python
from addernet import AdderNetLayer

rede = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)

celsius    = [0, 10, 20, 25, 30, 37, 50, 80, 100]
fahrenheit = [32, 50, 68, 77, 86, 98.6, 122, 176, 212]

rede.train(celsius, fahrenheit)

print(rede.predict(37))    # 98.60
print(rede.predict(100))   # 212.00
```

### Previsão em lote (numpy)

```python
import numpy as np

entradas = np.linspace(-50, 200, 1_000_000, dtype=np.float64)
saidas = rede.predict_batch(entradas)   # ~178M pred/s com AVX2
```

### Salvar e carregar

```python
rede.save("modelo.bin")
rede = AdderNetLayer.load("modelo.bin")
print(rede.predict(37))   # 98.60
```

---

## Uso — AdderNetHDC (múltiplas variáveis)

```python
from addernet import AdderNetHDC
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = MinMaxScaler(feature_range=(0, 150)).fit_transform(iris.data)
y = iris.target

model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)

# Treino single-pass (OnlineHD)
model.train(X, y)

# Retreino iterativo (AdaptHD) — melhora acurácia
model.train(X, y, n_iter=20, lr=1.0)

# Inferência em lote com cache
model.warm_cache()
preds = model.predict_batch(X)

print(f"Acurácia: {model.accuracy(X, y)*100:.1f}%")

model.save("iris.bin")
model = AdderNetHDC.load("iris.bin")
```

### Parâmetros de treino

| Parâmetro | Efeito |
|---|---|
| `n_iter=0` | Só OnlineHD (single-pass, mais rápido) |
| `n_iter=20, lr=1.0` | Bom padrão para maioria dos datasets |
| `n_iter=50, lr=0.5` | Conservador, para datasets pequenos |

### Generalização e criatividade

```python
# Misturar dois conceitos — gera novo hipervector sem ver dados novos
novo_hv = model.bundle_classes([0, 1])
classe  = model.classify_hv(novo_hv)

# Variações controladas por temperatura
for temp in [0.0, 0.1, 0.2, 0.3, 0.5]:
    variacao = model.add_noise(model.codebook[0], temp)
    print(f"temp={temp} → classe {model.classify_hv(variacao)}")
```

---

## Uso — AdderCluster (ensemble multi-nó)

```python
from addernet import AdderCluster
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = MinMaxScaler(feature_range=(0, 150)).fit_transform(iris.data)
y = iris.target

cluster = AdderCluster(
    n_nodes=4,
    strategy='feature',    # 'random' | 'range' | 'feature' | 'boosting'
    combination='vote',    # 'vote' | 'mean' | 'stack'
    input_min=0,
    input_max=150,
)

cluster.fit(X, y)
preds = cluster.predict_batch(X)

cluster.info()
```

---

## Uso — AdderBoost (gradient boosting sem multiplicação)

```python
from addernet import AdderBoost
import numpy as np

# Regressão: prever distância de frenagem dado velocidade
X = np.array([[v] for v in range(10, 110, 10)], dtype=np.float64)
y = (X[:, 0] ** 2 / 20).astype(np.float64)   # física simplificada

boost = AdderBoost(
    n_estimators=20,
    learning_rate=0.1,
    size=256, bias=50, input_min=0, input_max=110, lr=0.05
)
boost.fit(X, y, verbose=False)

preds = boost.predict_batch(X)
print(preds)
```

---

## Otimizações disponíveis

```python
from addernet import hdc_detect_backend

print(hdc_detect_backend())   # 'AVX2', 'NEON', ou 'SCALAR'

model.set_threads(4)      # multithreading (AdderNetHDC)
model.warm_cache()        # pré-computar hipervectors
model.set_cache(False)    # desligar cache (hardware com pouca RAM)
```

---

## Benchmarks (v1.0.8 · AVX2 · x86-64)

> Medidos com `time.perf_counter()` em ambiente Linux, Python 3.x.
> Splits treino/teste via `train_test_split(test_size=0.3, random_state=42)`.

### Benchmark 1 — AdderNetLayer: Celsius → Fahrenheit

| Métrica | Valor |
|---|---|
| Amostras | 1 000 000 |
| Tempo total | 5.63 ms |
| **Throughput** | **~178M pred/s** |
| MAE | 0.0000 °F |
| Multiplicações em inferência | **0** |

### Benchmark 2 — AdderNetHDC: Iris (4 vars, 3 classes)

| Métrica | Valor |
|---|---|
| Treino (n_iter=20) | 22 ms |
| Inferência (45 amostras) | 0.65 ms |
| **Throughput** | **~69 000 pred/s** |
| Acurácia (test set) | 86.7% |
| Multiplicações em inferência | **0** |

### Benchmark 3 — AdderNetHDC: Wine (13 vars, 3 classes)

| Métrica | Valor |
|---|---|
| Treino (n_iter=20) | 33 ms |
| Inferência (54 amostras, cache quente) | 1.68 ms |
| **Throughput** | **~32 000 pred/s** |
| Acurácia (test set) | 68.5% |
| Multiplicações em inferência | **0** |

### Benchmark 4 — AdderNetHDC: Breast Cancer (30 vars, 2 classes)

| Métrica | Valor |
|---|---|
| Treino (n_iter=20) | 303 ms |
| Inferência (171 amostras, cache quente) | 9.40 ms |
| **Throughput** | **~18 000 pred/s** |
| Acurácia (test set) | 85.4% |
| Multiplicações em inferência | **0** |

### Benchmark 5 — AdderNetLayer: Save / Load latency

| Operação | Latência |
|---|---|
| `save()` | 0.37 ms |
| `load()` | 0.10 ms |
| Predict pós-reload | correto (98.60 °F) |

---

## Casos de uso

- Sensores industriais com hardware sem FPU (ESP32, STM32)
- Classificação em tempo real embarcada
- Privacidade: dados originais não precisam ser guardados pós-treino
- One-shot learning: aprende classe nova com uma única amostra

---

## Exemplos prontos

```bash
# Celsius → Fahrenheit (uma variável)
python3 examples/basic/celsius_fahrenheit.py

# AdderNet-HDC com Iris
python3 examples/hdc/iris_hdc.py

# Benchmark HDC
python3 examples/hdc/benchmark_hdc.py
```

---

## Estrutura de arquivos

```
AdderNet/
├── src/              ← código C otimizado (AVX2, NEON, SCALAR)
├── python/           ← bindings Python (ctypes)
├── addernet/         ← pacote instalável
│   ├── addernet.py   ← AdderNetLayer
│   ├── addernet_hdc.py ← AdderNetHDC / AnHdcModel
│   ├── cluster.py    ← AdderCluster
│   └── boost.py      ← AdderBoost
├── tests/
├── examples/
│   ├── basic/        ← AdderNet básica
│   └── hdc/          ← AdderNet-HDC
├── Makefile          ← detecção automática de plataforma
└── pyproject.toml
```

---

## Limitações

- **AdderNetLayer**: apenas uma variável de entrada por camada
- **AdderNetHDC**: acurácia inferior a MLPs profundas em datasets complexos (troca por zero multiplicação)
- D muito pequeno (< 1000) pode colapsar a acurácia
- Contexto sequencial (LLM embarcado) ainda em desenvolvimento

---

## Licença

[Apache 2.0](LICENSE) — © Pedro Henrique Batista Silva
