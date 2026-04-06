# AdderGPT-2 v1.4.1 — Kaggle Setup (2× T4, 30GB RAM)

> **Código completo e corrigido** — build com `install-libs`, progresso visível, compatível com CUDA 12.8 + T4 sm_75.

```python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  ADDERGPT-2 v1.4.1 — Kaggle | 2× T4 | CUDA 12.8 | sm_75       ║
# ║  Build: make install-libs → sys.path → Treino com checkpoint   ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, sys, json, time, gc, subprocess, shutil, math
import numpy as np
from pathlib import Path

# ───────────────────────────────────────────────────────────────────
# 1. CONFIGURAÇÕES
# ───────────────────────────────────────────────────────────────────
SAMPLES         = 200_000    # janelas de contexto para treino
N_ITER          = 40         # épocas máximas
HV_DIM          = 2048       # dimensionalidade hiperdimensional
CONTEXT_SIZE    = 32         # tokens de contexto
VOCAB_SIZE      = 8000       # tamanho do vocabulário BPE

USE_GPU_INFERENCE = True     # inferência batch na T4
USE_GPU_TRAINING  = True     # AdaptHD/RefineHD na T4
CHECKPOINT_EVERY = 5         # salvar checkpoint a cada N épocas
KEEP_LAST_N      = 3         # manter apenas últimos N checkpoints
RESUME_FROM_CKPT = True      # retomar do último checkpoint se existir

KAGGLE_DIR = Path("/kaggle/working/AdderNet")
MODEL_DIR  = Path("/kaggle/working/addergpt_model")
CKPT_DIR   = MODEL_DIR / "checkpoints"
BUILD_DIR  = KAGGLE_DIR / "build"

MODEL_DIR.mkdir(exist_ok=True, parents=True)
CKPT_DIR.mkdir(exist_ok=True, parents=True)

VERBOSE_LEVEL = 5  # imprime log a cada N épocas

# ───────────────────────────────────────────────────────────────────
# 2. CLONE + BUILD + INSTALL-LIBS
# ───────────────────────────────────────────────────────────────────
print("🔧 [Setup] AdderNet v1.4.1 — CUDA 12.8 + T4 sm_75")

if not KAGGLE_DIR.exists():
    print("📦 Clonando AdderNet...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/PedroHenriqueBatistaSilva/AdderNet.git",
         str(KAGGLE_DIR)],
        check=True
    )

# Ambiente CUDA
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:/usr/local/nvidia/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ── Build CPU ──
print("🛠️  Compilando bibliotecas base (CPU)...")
res = subprocess.run(
    ["make", "-C", str(KAGGLE_DIR), "all"],
    capture_output=True, text=True
)
if res.returncode != 0:
    print(f"❌ Falha no make all:\n{res.stderr[-500:]}")
    raise RuntimeError("Compilação CPU falhou")
print("✅ Bibliotecas CPU compiladas")

# ── Build CUDA 2026 ──
cuda_so = BUILD_DIR / "libaddernet_cuda_2026.so"
if not cuda_so.exists():
    print("🚀 Compilando CUDA 2026 (T4 sm_75)...")
    res = subprocess.run(
        ["make", "-C", str(KAGGLE_DIR), "cuda_2026"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ}
    )
    if res.returncode == 0 and cuda_so.exists():
        print("✅ CUDA 2026 compilado")
    else:
        print(f"⚠️  CUDA indisponível (rc={res.returncode}). Fallback CPU.")
        if res.stderr:
            print(f"   stderr: {res.stderr[-500:]}")
        USE_GPU_INFERENCE = USE_GPU_TRAINING = False

# ── Install-libs: copia .so de build/ para addernet/ ──
print("🔗 Instalando libs no pacote Python...")
res2 = subprocess.run(
    ["make", "-C", str(KAGGLE_DIR), "install-libs"],
    capture_output=True, text=True
)
if res2.returncode == 0:
    print(f"   {res2.stdout.strip()}")
else:
    print("   ⚠️  install-libs falhou, copy manual...")
    for lib in BUILD_DIR.glob("*.so"):
        shutil.copy2(lib, KAGGLE_DIR / "addernet" / lib.name)

# ── Deps Python ──
print("📦 Instalando dependências Python...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "tokenizers", "datasets", "psutil"],
    capture_output=True, text=True
)

# ── Import ──
sys.path.insert(0, str(KAGGLE_DIR))
try:
    from addernet import AdderNetHDC, AdderAttention
    try:
        from addernet import hdc_detect_backend
    except ImportError:
        def hdc_detect_backend(): return "CPU"
    print("✅ AdderNet importado com sucesso!")
except OSError as e:
    print(f"❌ Falha ao carregar bibliotecas: {e}")
    print("💡 Listando addernet/:")
    for f in (KAGGLE_DIR / "addernet").glob("*.so"):
        print(f"   {f.name} ({f.stat().st_size} bytes)")
    raise

def now(): return time.strftime("%H:%M:%S")
def ram_gb(): return psutil.virtual_memory().used / (1024**3)

print("\n" + "="*70)
print(f"  [{now()}] 🧠 AdderGPT-2 v1.4.1 | Kaggle + 2× T4")
print(f"  Backend: {hdc_detect_backend()} | HV_DIM: {HV_DIM}")
print(f"  GPU Train: {USE_GPU_TRAINING} | GPU Infer: {USE_GPU_INFERENCE}")
print("="*70 + "\n")

# ───────────────────────────────────────────────────────────────────
# 3. TOKENIZER & DATASET
# ───────────────────────────────────────────────────────────────────
print(f"[{now()}] [1/5] Carregando Wikitext-103...")
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

tok_path = MODEL_DIR / "tokenizer.json"
if not tok_path.exists():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    def batch_iterator(bs=1000):
        for i in range(0, len(dataset), bs):
            yield dataset[i:i+bs]["text"]
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(str(tok_path))
    print(f"  ✅ Tokenizer treinado: {VOCAB_SIZE} tokens")
else:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"  ✅ Tokenizer carregado de {tok_path}")

BOS, EOS, PAD = [tokenizer.token_to_id(t) for t in ["[BOS]", "[EOS]", "[PAD]"]]

# ───────────────────────────────────────────────────────────────────
# 4. EMBEDDINGS & FEATURE EXTRACTION
# ───────────────────────────────────────────────────────────────────
print(f"[{now()}] [2/5] Gerando embeddings HDC ({VOCAB_SIZE} × {HV_DIM})...")
emb_path = MODEL_DIR / "vocab_embeddings.npy"
if not emb_path.exists():
    rng = np.random.default_rng(42)
    vocab_emb = rng.choice([-1, 1], size=(VOCAB_SIZE, HV_DIM)).astype(np.float32)
    np.save(emb_path, vocab_emb)
    print(f"  ✅ Embeddings gerados e salvos")
else:
    vocab_emb = np.load(emb_path)
    print(f"  ✅ Embeddings carregados de {emb_path}")

print(f"[{now()}] [3/5] Extraindo {SAMPLES} janelas com AdderAttention...")
attention = AdderAttention(threshold=None)
X_train = np.zeros((SAMPLES, HV_DIM), dtype=np.float64)
y_train = np.zeros((SAMPLES,), dtype=np.int32)

sample_idx = 0
bar_width = 50
for row in dataset:
    if sample_idx >= SAMPLES:
        break
    text = row["text"].strip()
    if not text:
        continue
    ids = tokenizer.encode(text).ids
    if len(ids) < 2:
        continue
    for i in range(1, len(ids)):
        if sample_idx >= SAMPLES:
            break
        window = ids[max(0, i - CONTEXT_SIZE):i]
        if len(window) < CONTEXT_SIZE:
            window = [BOS] * (CONTEXT_SIZE - len(window)) + window
        ctx_vecs = vocab_emb[window]
        Q = ctx_vecs[-1:].reshape(1, 1, HV_DIM)
        K = ctx_vecs.reshape(1, CONTEXT_SIZE, HV_DIM)
        V = K.copy()
        context_vec = attention.forward(Q, K, V).squeeze()
        x_norm = ((context_vec / CONTEXT_SIZE) + 1.0) / 2.0 * 255.0
        X_train[sample_idx] = np.clip(np.nan_to_num(x_norm), 0, 255).astype(np.float64)
        y_train[sample_idx] = ids[i]
        sample_idx += 1

# Barra de progresso
done = int(sample_idx / SAMPLES * bar_width)
bar = "█" * done + "░" * (bar_width - done)
print(f"\r  [{bar}] {sample_idx}/{SAMPLES} amostras")

del dataset, tokenizer, vocab_emb, attention
gc.collect()
print(f"  → RAM: {ram_gb():.1f} GB | Amostras: {sample_idx}")

# ───────────────────────────────────────────────────────────────────
# 5. CHECKPOINT & TREINO COM PROGRESSO VISÍVEL
# ───────────────────────────────────────────────────────────────────
def save_checkpoint(model, epoch, val_acc, path):
    model.save(str(path / "model.bin"))
    with open(path / "meta.json", "w") as f:
        json.dump({
            "epoch": epoch,
            "val_accuracy": float(val_acc),
            "timestamp": now()
        }, f, indent=2)
    print(f"  💾 Checkpoint salvo: epoch={epoch}, acc={val_acc:.2%}")

def load_checkpoint(path):
    if not (path / "model.bin").exists():
        return None, 0
    try:
        with open(path / "meta.json") as f:
            meta = json.load(f)
        model = AdderNetHDC.load(str(path / "model.bin"))
        return model, meta["epoch"]
    except Exception as e:
        print(f"  ⚠️  Erro ao carregar checkpoint: {e}")
        return None, 0

def cleanup_old(ckpt_dir, keep_n):
    ckpts = sorted(
        ckpt_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1])
    )
    while len(ckpts) > keep_n:
        shutil.rmtree(ckpts.pop(0))

def print_progress_bar(epoch, total, acc, best_acc, elapsed, ram):
    """Imprime barra de progresso + métricas do treinamento."""
    pct = epoch / total
    bar_w = 40
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    mins = elapsed / 60
    eta_min = (elapsed / max(epoch, 1)) * (total - epoch) / 60

    print(f"  [{bar}] {epoch}/{total} | "
          f"Acc: {acc:5.2%} | Best: {best_acc:5.2%} | "
          f"⏱ {mins:.1f}m | ETA: {eta_min:.1f}m | "
          f"RAM: {ram:.1f}GB")

# ── Preparar dados ──
print(f"\n[{now()}] [4/5] Iniciando treino...")
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.int32)

start_epoch = 0
if RESUME_FROM_CKPT:
    last = sorted(
        CKPT_DIR.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1])
    )
    if last:
        ckpt_model, start_epoch = load_checkpoint(last[-1])
        if ckpt_model is not None:
            model = ckpt_model
            print(f"  🔄 Retomando do checkpoint: epoch {start_epoch}")

if start_epoch == 0:
    print(f"  🆕 Criando AdderNetHDC (n_vars={HV_DIM}, classes={VOCAB_SIZE}, hv_dim={HV_DIM})...")
    model = AdderNetHDC(
        n_vars=HV_DIM,
        n_classes=VOCAB_SIZE,
        table_size=256,
        hv_dim=HV_DIM,
        use_gpu=USE_GPU_INFERENCE,
        use_gpu_training=USE_GPU_TRAINING,
        seed=42
    )
    model.set_threads(4)
    print(f"  ✅ Modelo criado | GPU Train: {USE_GPU_TRAINING} | GPU Infer: {USE_GPU_INFERENCE}")

start_time = time.time()
best_acc, no_improve, patience = 0.0, 0, 15
global_epoch = start_epoch  # track total epochs including resumed

try:
    for epoch in range(start_epoch, N_ITER):
        epoch_start = time.time()

        # Treina 1 iteração (AdaptHD na GPU)
        history = model.train(
            X=X_train, y=y_train,
            n_iter=1, lr=1.0, margin='5%',
            regenerate=0.02, patience=0, interactions=10,
            verbose=0  # silencioso — nós imprimimos a barra
        )

        global_epoch = epoch + 1
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        curr_acc = history.get('best_val_accuracy', 0.0)

        # Atualiza best_acc
        if curr_acc > best_acc:
            best_acc = curr_acc
            no_improve = 0
        else:
            no_improve += 1

        # Progresso visível
        print_progress_bar(
            global_epoch, N_ITER,
            curr_acc, best_acc,
            total_elapsed, ram_gb()
        )

        # Checkpoint
        if global_epoch % CHECKPOINT_EVERY == 0:
            ckpt = CKPT_DIR / f"epoch_{global_epoch:03d}"
            ckpt.mkdir(exist_ok=True)
            save_checkpoint(model, global_epoch, curr_acc, ckpt)
            cleanup_old(CKPT_DIR, KEEP_LAST_N)
            model.save(str(MODEL_DIR / "adder_lm_model.bin"))

        # Early stopping
        if no_improve >= patience:
            print(f"\n  ⏹️  Early stopping após {patience} épocas sem melhoria")
            print(f"  🏆 Melhor acurácia: {best_acc:.2%} @ epoch {global_epoch - no_improve}")
            break

except KeyboardInterrupt:
    print(f"\n  ⚠️  Interrompido pelo usuário na época {global_epoch}")
    model.save(str(MODEL_DIR / "adder_lm_model.bin"))
except Exception as e:
    print(f"\n  ❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    model.save(str(MODEL_DIR / "adder_lm_model.bin"))

end_time = time.time()

# ───────────────────────────────────────────────────────────────────
# 6. SALVAMENTO FINAL
# ───────────────────────────────────────────────────────────────────
print(f"\n[{now()}] [5/5] Salvando modelo final...")
model.save(str(MODEL_DIR / "adder_lm_model.bin"))

cfg = {
    "version": "AdderGPT-2 v1.4.1-Kaggle",
    "epochs": global_epoch,
    "best_acc": float(best_acc),
    "hv_dim": HV_DIM,
    "vocab_size": VOCAB_SIZE,
    "context_size": CONTEXT_SIZE,
    "gpu": {
        "train": USE_GPU_TRAINING,
        "infer": USE_GPU_INFERENCE
    }
}
with open(MODEL_DIR / "config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# Lista arquivos salvos
print(f"\n  📁 Conteúdo de {MODEL_DIR}:")
for p in sorted(MODEL_DIR.rglob("*")):
    if p.is_file():
        sz = p.stat().st_size / 1024
        print(f"     {p.relative_to(MODEL_DIR)} ({sz:.0f} KB)")

# Zip para download
zip_path = MODEL_DIR.parent / "addergpt_model.zip"
print(f"\n  🗜️  Compactando...")
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', MODEL_DIR)
zip_sz = zip_path.stat().st_size / (1024**2)
print(f"  📦 {zip_path.name} ({zip_sz:.1f} MB)")

total_h = (end_time - start_time) / 3600
print(f"\n{'='*70}")
print(f"  🏁 Concluído em {total_h:.2f}h | Acc: {best_acc:.2%} | Épocas: {global_epoch}")
print(f"  📦 Download: {zip_path}")
print(f"{'='*70}")
```

## O que mudou vs. sua versão antiga

| Problema | Antes | Agora |
|---|---|---|
| **Build** | `cp *.so /usr/local/lib` manual | `make install-libs` (copia para `addernet/`) |
| **`load_dataset`** | Não importado → NameError | `from datasets import load_dataset` |
| **Progresso** | Só log do C a cada 5 épocas | Barra de progresso + ETA + RAM por época |
| **`verbose=0`** no train | Silencia log interno do C | Você controla o output — barra limpa |
| **Checkpoint resume** | Carrega modelo mas não seta | Set `model = ckpt_model` corretamente |
| **`global_epoch`** | Contagem resetava no resume | Contagem contínua para barra correta |
| **ETA** | Não existia | Calculado por `(tempo/época) × épocas restantes` |
| **Zip listing** | Não listava arquivos | Lista tudo que foi salvo |

## O que esperar no Kaggle

- **Build**: ~30-60s (CPU) + ~60-120s (CUDA 2026)
- **Feature extraction**: ~5-15min (200k amostras com AdderAttention)
- **Treino**: depende de `N_ITER` — ~1-3min/época na T4 com GPU training
- **VRAM**: ~260MB por época (HV_DIM=2048, VOCAB_SIZE=8000)
- **RAM**: pico ~10-15GB durante feature extraction (GC libera depois)
