# 🔬 Grokking — Mechanistic Interpretability Project

An interactive Streamlit app that reproduces and extends the grokking experiments from:

> **Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023)**  
> https://arxiv.org/abs/2301.05217

Grokking is a surprising phenomenon where a neural network first **memorises** a training set (achieving 100% train accuracy but near-random validation accuracy), then — much later, under the pressure of weight decay — suddenly **generalises** to perfect validation accuracy. This project trains a 1-layer Transformer on modular arithmetic and uses Fourier analysis to reveal *why* and *how* this happens.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch streamlit matplotlib einops numpy
```

> **Note:** `einops` will also auto-install itself if missing when the app runs.

### 2. Run the app

```bash
streamlit run Projectgrokking.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## ⚙️ Hardware

| Hardware | Experience |
|----------|-----------|
| NVIDIA GPU (CUDA) | ✅ Recommended — training is fast, a green GPU badge appears in the app |
| CPU only | ⚠️ Works, but training 20,000 epochs will be slow (expect several minutes per run) |

The app auto-detects your hardware and adjusts accordingly. Results may differ very slightly between GPU and CPU runs due to floating-point precision differences, but the overall behaviour (grokking occurring, Fourier structure, etc.) will be the same.

---

## 🧭 App Overview

The sidebar lets you configure global hyperparameters that apply to all components:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prime modulus `p` | 97 | The modulus for arithmetic operations |
| Training fraction | 0.30 | Fraction of all p² pairs used for training |
| Epochs | 20,000 | Total training steps |
| Learning rate | 1e-3 | AdamW learning rate |
| Weight decay | 2.0 | Key driver of grokking — higher = faster generalisation |
| `d_model` | 128 | Transformer hidden dimension |
| Attention heads | 4 | Number of attention heads |
| Random seed | 42 | Controls reproducibility |

The main area is divided into **7 tabs**, each a self-contained experiment:

---

### ① Training & Report
Trains a 1-layer Transformer on **(a + b) mod p** and plots loss and accuracy curves. Detects the grokking epoch (when validation accuracy first crosses 95%) and generates a written training report explaining the result.

### ② Fourier Analysis
Applies a Discrete Fourier Transform to the embedding matrix and MLP neuron activations. Shows that the model maps inputs to a **sparse set of key frequencies** — the fingerprint of the Fourier circuit the model has learned.

> Requires Component 1 to be run first.

### ③ Three Phases
Computes **excluded loss**, **restricted loss**, and **weight norm** across saved checkpoints to chart the three distinct phases of grokking:
1. **Memorisation** — high-norm lookup table, no Fourier circuit
2. **Circuit formation** — Fourier circuit builds in parallel with memorisation
3. **Cleanup / Grokking** — weight decay removes memorisation, only the Fourier circuit remains

> Requires Component 1 to be run first.

### ④ Data-Fraction Study
Trains 9 separate models across training fractions from 10% to 90% and plots how the grokking epoch varies. Demonstrates that grokking is a **data-scarcity phenomenon**.

### ⑤ AI Alignment Essay
A long-form essay on why **mechanistic interpretability** is important for AI alignment — covering behavioural testing limitations, superposition, circuit-level verification, deceptive alignment, and open research challenges.

### ⑥ Modular Multiplication
Trains on **(a × b) mod p** and analyses the learned algorithm via Fourier analysis and **discrete logarithm theory** — showing that multiplication reduces to addition in the discrete-log domain.

### ⑦ Co-Grokking
Trains a **single model on both addition and multiplication simultaneously** and demonstrates *co-grokking*: both tasks generalise at approximately the same epoch, because they share a common embedding representation.

---

## 📁 File Structure

```
Projectgrokking.py   ← the entire app (single file, no other files needed)
README.md            ← this file
```

All datasets are **generated programmatically** — no external data files are required.

---

## 🔁 Reproducibility

Results are reproducible across runs **on the same hardware type** (GPU or CPU) when using the same random seed. The seed controls both `torch.manual_seed` and Python's `random.seed`.

---

## 📚 Background Reading

- [Nanda et al. (2023) — Progress Measures for Grokking](https://arxiv.org/abs/2301.05217)
- [Power et al. (2022) — Grokking: Generalization Beyond Overfitting](https://arxiv.org/abs/2201.02177)
- [Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

---

## 🧩 Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Neural network, training, tensor operations |
| `streamlit` | Interactive web UI |
| `matplotlib` | All charts and visualisations |
| `einops` | Tensor reshaping in the Transformer |
| `numpy` | Numerical operations, FFT for Fourier analysis |
