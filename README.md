# AdderNet

[![PyPI version](https://img.shields.io/pypi/v/addernet.svg)](https://pypi.org/project/addernet/)
[![Python](https://img.shields.io/pypi/pyversions/addernet.svg)](https://pypi.org/project/addernet/)
[![License](https://img.shields.io/github/license/PedroHenriqueBatistaSilva/AdderNet.svg)](LICENSE)

Biblioteca de machine learning que **não usa multiplicação de ponto flutuante** na inferência. Zero.

> Benchmarks medidos em CPU x86-64 com backend **AVX2** e GPUs NVIDIA via **CUDA 2026**, Python 3.x, v1.4.0.

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
| `AdderAttention` | Attention mechanism baseado em distância de Hamming — zero multiplicação |

---

## Novidades v1.4.0 🚀

### CUDA 2026 — Modernização Completa
- **Kernel 2026 Ampere+**: Treinamento cooperativo com shared memory 100KB, warp-level primitives, e unified kernel (encode → Hamming → update em um único launch)
- **Kernel Selection Automático**: Detecta a GPU e seleciona o kernel otimizado (Ampere sm_80+ → Turing sm_70-75 → Legacy sm_61)
- **Unified Memory**: Zero-copy GPU memory para datasets pequenos (ativa com `ADDERNET_UNIFIED_MEMORY=1`)
- **CUDA Graphs**: Capture once, replay many (ativa com `ADDERNET_CUDA_GRAPHS=1`)
- **Persistent Kernel**: Elimina overhead de kernel launch (ativa com `ADDERNET_PERSISTENT_KERNEL=1`)

### AdderAttention — Attention Mechanism sem Multiplicação
- Mecanismo de atenção baseado em distância de Hamming
- Ideal para transformers-like architectures em embedded systems
- Sem operações de ponto flutuante

### Recursos Mantidos
- **HV_DIM Dinâmico**: Dimensionalidade hiperdimensional configurável em runtime (`512`, `1024`, `2048`, `4096`, etc)
- **Aceleração CUDA no Treinamento**: AdaptHD/RefineHD paralelo em GPU com `atomicAdd`
- **Aceleração CUDA na Inferência**: `predict_batch` via kernels CUDA dedicados
- **Compatibilidade e Fallback**: Fallback automático para CPU (AVX2/NEON/SCALAR) quando GPU não disponível

---

## Instalação

```bash
pip install addernet
```

Ou do código-fonte (para compilar com otimizações nativas e CUDA opcional):

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
make all         # Compila binários da CPU
make cuda_native # Opcional: Compila o backend de GPU (requer nvcc)
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

---

## Uso — AdderNetHDC (Aceleração GPU e HDC Dinâmico)

```python
from addernet import AdderNetHDC
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = MinMaxScaler(feature_range=(0, 150)).fit_transform(iris.data)
y = iris.target

# HV_DIM dinâmico configurável (ex: 2048, 4096)
model = AdderNetHDC(
    n_vars=4, 
    n_classes=3, 
    table_size=256, 
    hv_dim=4096,              # <- Dimensionalidade configurável no runtime!
    use_gpu=True,             # <- Ativa inferência batch em CUDA
    use_gpu_training=True     # <- Ativa treinamento iterativo em CUDA
)

# Arrays numpy precisam ser "C Contiguous" 
X_c = np.ascontiguousarray(X, dtype=np.float64)
y_c = np.ascontiguousarray(y, dtype=np.int32)

# Treino single-pass (OnlineHD)
model.train(X_c, y_c)

# Retreino iterativo (AdaptHD) — massivamente paralelo na GPU
model.train(X_c, y_c, n_iter=20, lr=1.0)

# Inferência massiva e ultrarrápida via GPU
preds = model.predict_batch(X_c)

print(f"Acurácia: {model.accuracy(X_c, y_c)*100:.1f}%")
```

---

## Uso — AdderCluster (ensemble multi-nó)

```python
from addernet import AdderCluster
import numpy as np

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

## Uso — AdderAttention (Attention sem Multiplicação)

```python
from addernet import AdderAttention
import numpy as np

# Attention mechanism baseado em distância de Hamming
# Ideal para embedded systems sem FPU
attn = AdderAttention(
    n_vars=4,
    n_heads=8,  # Número de heads de atenção
    hv_dim=2048  # Dimensionalidade dos hipervetores
)

# Query, Key, Value (em formato hypervector)
# Treinamento
attn.fit(X_train, y_train)

# Attention scores usando Hamming distance
scores = attn.attention(X_query, X_keys)

# Classificação com attention
prediction = attn.predict(X_test)
```

### Configurações Avançadas do CUDA 2026

```python
import os
os.environ['ADDERNET_UNIFIED_MEMORY'] = '1'   # Zero-copy GPU memory
os.environ['ADDERNET_CUDA_GRAPHS'] = '1'       # Capture/replay
os.environ['ADDERNET_PERSISTENT_KERNEL'] = '1' # Elimina kernel launch overhead
```

---

## Otimizações disponíveis

```python
from addernet import hdc_detect_backend

print(hdc_detect_backend())   # 'AVX2', 'NEON', ou 'SCALAR'

model.set_threads(4)      # multithreading CPU (AdderNetHDC)
model.warm_cache()        # pré-computar hipervectors
model.set_cache(False)    # desligar cache (hardware com pouca RAM)
```

---

## Limitações

- **AdderNetLayer**: apenas uma variável de entrada por camada
- **AdderNetHDC**: acurácia inferior a MLPs profundas em datasets complexos (troca por zero multiplicação)
- `hv_dim` muito pequeno (< 1000) pode colapsar a acurácia, use a Dimensionalidade Dinâmica para testar!

---

## Licença

[Apache 2.0](LICENSE) — © Pedro Henrique Batista Silva