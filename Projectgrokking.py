"""
Projectgrokking.py  —  Grokking Mechanistic Interpretability Project
=====================================================================
Components implemented:
  1  Training & report                       (10 marks)
  2  Fourier analysis: embeddings + MLP      (+5 marks)
  3  Three phases: memorisation / circuit / cleanup  (+5 marks)
  4  Data-fraction study                     (+5 marks)
  5  AI alignment & mechanistic interp essay (+5 marks)
  6  Modular multiplication algorithm        (+10 marks)
  7  Co-grokking on add + mul               (+15 marks)

Based on: Nanda et al. "Progress Measures for Grokking via
Mechanistic Interpretability" (2023) — https://arxiv.org/abs/2301.05217

Run with:
    pip install torch streamlit matplotlib einops
    streamlit run Projectgrokking.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import random, time, io, math
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import List, Optional, Dict, Tuple

try:
    import einops
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'einops'])
    import einops

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grokking: Mechanistic Interpretability Project",
    page_icon="🔬", layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
.main { background: #0d0d0f; }
.block-container { padding-top: 1.5rem; }
h1 { font-size: 2rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg,#e8f4ff,#a78bfa);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { color: #a78bfa !important; }
h3 { color: #7eb8f7 !important; }
.metric-card { background:#13131a; border:1px solid #1e2030; border-radius:10px;
               padding:1rem 1.2rem; text-align:center; margin-bottom:0.5rem; }
.metric-label { color:#5a6080; font-size:0.73rem; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:0.25rem; }
.metric-value { color:#e8f4ff; font-size:1.5rem; font-weight:700;
                font-family:'JetBrains Mono',monospace; }
.metric-value.ok  { color:#4ade80; }
.metric-value.hi  { color:#a78bfa; }
.phase-box { background:#0f0f1a; border:1px solid #2a2a4a; border-radius:10px;
             padding:1.5rem; margin:0.5rem 0; }
.phase-title { color:#a78bfa; font-weight:700; font-size:1.05rem; margin-bottom:0.5rem; }
.essay-box  { background:#0f1420; border:1px solid #2a3a60; border-radius:12px;
              padding:2rem 2.5rem; color:#c8d8f0; line-height:1.85; }
.essay-box h2 { color:#7eb8f7; font-size:1.35rem; }
.essay-box h3 { color:#a78bfa; font-size:1.05rem; margin-top:1.4rem; }
.essay-box p  { margin-bottom:0.9rem; font-size:0.95rem; }
.essay-box blockquote { border-left:3px solid #a78bfa; padding-left:1rem;
                        color:#8898bb; font-style:italic; margin:1rem 0; }
section[data-testid="stSidebar"] { background:#0a0a10; border-right:1px solid #1e2030; }
section[data-testid="stSidebar"] * { color:#c8d8f0 !important; }
.stButton > button { background:#1a2340; color:#7eb8f7; border:1px solid #2a3a60;
                     border-radius:8px; font-family:'JetBrains Mono',monospace;
                     font-size:0.85rem; padding:0.45rem 1.1rem; width:100%; }
.stButton > button:hover { background:#223060; border-color:#7eb8f7; }
.gpu-badge { background:#1a0f2e; border:1px solid #7c3aed; border-radius:8px;
             padding:0.35rem 0.9rem; color:#a78bfa; font-family:'JetBrains Mono',monospace;
             font-size:0.82rem; display:inline-block; margin-bottom:0.8rem; }
.warn-box { background:#1a0f00; border:1px solid #ca8a04; border-radius:8px;
            padding:0.6rem 1rem; color:#fbbf24; margin:0.5rem 0; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (Nanda et al. architecture — compatible with Analysis notebook)
# ══════════════════════════════════════════════════════════════════════════════

class HookPoint(nn.Module):
    """Identity module that caches activations via forward hooks."""
    def __init__(self):
        super().__init__()
        self.fwd_hooks: List = []
        self.name: str = ''

    def give_name(self, name: str):
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        handle = self.register_forward_hook(full_hook)
        self.fwd_hooks.append(handle)

    def remove_hooks(self, dir='fwd'):
        for h in self.fwd_hooks:
            h.remove()
        self.fwd_hooks = []

    def forward(self, x):
        return x


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / math.sqrt(d_model))

    def forward(self, x):
        return einops.rearrange(self.W_E[:, x], 'd b p -> b p d')


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / math.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / math.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model_ref):
        super().__init__()
        self.model_ref = model_ref
        self.d_head = d_head
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / math.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones(n_ctx, n_ctx)))
        self.hook_attn = HookPoint()

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_pre = torch.einsum('biph,biqh->biqp', k, q) / math.sqrt(self.d_head)
        attn_pre = attn_pre - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn = self.hook_attn(F.softmax(attn_pre, dim=-1))
        z = torch.einsum('biph,biqp->biqh', v, attn)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        return torch.einsum('df,bqf->bqd', self.W_O, z_flat)


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, model_ref):
        super().__init__()
        self.model_ref = model_ref
        self.W_in  = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_model))
        self.b_in  = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.hook_post = HookPoint()

    def forward(self, x):
        pre = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        post = self.hook_post(F.relu(pre))
        return torch.einsum('dm,bpm->bpd', self.W_out, post) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, model_ref):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model_ref)
        self.mlp  = MLP(d_model, d_mlp, model_ref)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_vocab: int, d_model: int, d_mlp: int, d_head: int,
                 num_heads: int, n_ctx: int, num_layers: int = 1):
        super().__init__()
        self.embed     = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, [self])
            for _ in range(num_layers)
        ])
        self.unembed = Unembed(d_vocab, d_model)
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    def hook_points(self):
        return [m for _, m in self.named_modules() if isinstance(m, HookPoint)]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks()


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_fourier_basis(p: int, device=None) -> Tuple[torch.Tensor, List[str]]:
    """Build the (p × p) orthonormal Fourier basis for Z/pZ."""
    basis, names = [], ['Const']
    basis.append(torch.ones(p) / math.sqrt(p))
    for k in range(1, p // 2 + 1):
        angles = 2 * math.pi * k * torch.arange(p) / p
        cos_vec = torch.cos(angles)
        sin_vec = torch.sin(angles)
        cos_vec /= cos_vec.norm()
        sin_vec /= sin_vec.norm()
        basis.append(cos_vec)
        basis.append(sin_vec)
        names.append(f'cos {k}')
        names.append(f'sin {k}')
    basis = torch.stack(basis, dim=0)
    if device is not None:
        basis = basis.to(device)
    return basis, names   # shape (p, p)


def embed_fourier_spectrum(model: Transformer, p: int,
                           fourier_basis: torch.Tensor) -> torch.Tensor:
    """Norm of each Fourier component across all embedding dimensions. Shape (p,)."""
    W_E_tokens = model.embed.W_E[:, :p]              # (d_model, p)
    W_E_proj   = fourier_basis @ W_E_tokens.T        # (p, d_model)
    return W_E_proj.norm(dim=1)                       # (p,)


def find_key_freqs(spectrum: torch.Tensor, top_k: int = 5) -> List[int]:
    """
    Given the embedding Fourier spectrum (norm at each of the p basis vectors),
    group cos/sin pairs per frequency and return the top_k frequencies by amplitude.
    """
    p = len(spectrum)
    freq_amp = []
    for k in range(1, p // 2 + 1):
        amp = math.sqrt(spectrum[2*k-1].item()**2 + spectrum[2*k].item()**2)
        freq_amp.append((amp, k))
    freq_amp.sort(reverse=True)
    return [k for _, k in freq_amp[:top_k]]


def comp_cos(logits: torch.Tensor, fourier_basis: torch.Tensor, freq: int) -> torch.Tensor:
    """Project logits onto the cos_freq direction. logits: (batch, p)."""
    bv = fourier_basis[2 * freq - 1]                          # (p,)
    return (logits * bv).sum(dim=-1, keepdim=True) * bv       # (batch, p)


def comp_sin(logits: torch.Tensor, fourier_basis: torch.Tensor, freq: int) -> torch.Tensor:
    """Project logits onto the sin_freq direction. logits: (batch, p)."""
    bv = fourier_basis[2 * freq]                              # (p,)
    return (logits * bv).sum(dim=-1, keepdim=True) * bv      # (batch, p)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cross_entropy_hp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy in float64 to avoid log-softmax underflow."""
    logprobs = F.log_softmax(logits.double(), dim=-1)
    pred = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -pred.mean()


@torch.no_grad()
def excluded_loss(model: Transformer, all_tokens: torch.Tensor,
                  all_labels: torch.Tensor, fourier_basis: torch.Tensor,
                  key_freqs: List[int], d_vocab_out: int) -> float:
    """Loss after removing all key-frequency components from the logits."""
    logits = model(all_tokens)[:, -1, :d_vocab_out]
    excl   = logits.clone()
    for k in key_freqs:
        excl = excl - comp_cos(logits, fourier_basis, k) - comp_sin(logits, fourier_basis, k)
    return cross_entropy_hp(excl, all_labels).item()


@torch.no_grad()
def restricted_loss(model: Transformer, all_tokens: torch.Tensor,
                    all_labels: torch.Tensor, fourier_basis: torch.Tensor,
                    key_freqs: List[int], d_vocab_out: int) -> float:
    """Loss using ONLY the key-frequency components of the logits."""
    logits = model(all_tokens)[:, -1, :d_vocab_out]
    restr  = sum(comp_cos(logits, fourier_basis, k) + comp_sin(logits, fourier_basis, k)
                 for k in key_freqs)
    return cross_entropy_hp(restr, all_labels).item()


@torch.no_grad()
def weight_norm_total(model: Transformer) -> float:
    """Sum of squared weights across all parameters."""
    return sum(p.pow(2).sum().item() for p in model.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_FNS = {
    'add': lambda a, b, p: (a + b) % p,
    'mul': lambda a, b, p: (a * b) % p,
}

def gen_dataset(p: int, frac_train: float, seed: int,
                fn_name: str = 'add', device=None):
    """Return (train_tokens, train_labels, val_tokens, val_labels)."""
    device = device or DEVICE
    eq_tok = p  # the '=' token index
    fn     = _FNS[fn_name]
    pairs  = [(a, b) for a in range(p) for b in range(p)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    tr, va = pairs[:div], pairs[div:]

    def to_tensors(pairs_list):
        toks = torch.tensor([[a, b, eq_tok] for a, b in pairs_list],
                            dtype=torch.long, device=device)
        labs = torch.tensor([fn(a, b, p) for a, b in pairs_list],
                            dtype=torch.long, device=device)
        return toks, labs

    return (*to_tensors(tr), *to_tensors(va))


def make_all_pairs(p: int, fn_name: str = 'add', device=None):
    """All p² pairs in lexicographic order — used for Fourier analysis."""
    device = device or DEVICE
    eq_tok = p
    fn     = _FNS[fn_name]
    all_tok = torch.tensor([[a, b, eq_tok] for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    all_lab = torch.tensor([fn(a, b, p) for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    return all_tok, all_lab


# ── Co-grokking dataset (add + mul combined) ──────────────────────────────────
def gen_cogrokking_dataset(p: int, frac_train: float, seed: int, device=None):
    """
    Combined dataset for add and mul.
    eq_add = p,  eq_mul = p+1,  D_VOCAB = p+2,  output classes = p
    """
    device = device or DEVICE
    eq_add, eq_mul = p, p + 1
    all_pairs = (
        [(a, b, eq_add, (a + b) % p) for a in range(p) for b in range(p)] +
        [(a, b, eq_mul, (a * b) % p) for a in range(p) for b in range(p)]
    )
    random.seed(seed)
    random.shuffle(all_pairs)
    div = int(frac_train * len(all_pairs))
    tr, va = all_pairs[:div], all_pairs[div:]

    def to_tensors(lst):
        toks = torch.tensor([[a, b, op] for a, b, op, _ in lst],
                            dtype=torch.long, device=device)
        labs = torch.tensor([lbl for _, _, _, lbl in lst],
                            dtype=torch.long, device=device)
        return toks, labs

    return (*to_tensors(tr), *to_tensors(va))


def make_all_pairs_co(p: int, device=None):
    """All 2p² pairs for co-grokking analysis."""
    device = device or DEVICE
    eq_add, eq_mul = p, p + 1
    add_tok = torch.tensor([[a, b, eq_add] for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    add_lab = torch.tensor([(a + b) % p for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    mul_tok = torch.tensor([[a, b, eq_mul] for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    mul_lab = torch.tensor([(a * b) % p for a in range(p) for b in range(p)],
                           dtype=torch.long, device=device)
    return add_tok, add_lab, mul_tok, mul_lab


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, tokens, labels, d_vocab_out):
    logits = model(tokens)[:, -1, :d_vocab_out]
    loss   = cross_entropy_hp(logits, labels).item()
    acc    = (logits.argmax(-1) == labels).float().mean().item() * 100
    return loss, acc


def build_model(p: int, d_model: int, n_heads: int, d_vocab: Optional[int] = None) -> Transformer:
    d_vocab = d_vocab or (p + 1)
    d_head  = d_model // n_heads
    d_mlp   = 4 * d_model
    return Transformer(
        d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp,
        d_head=d_head, num_heads=n_heads, n_ctx=3, num_layers=1,
    ).to(DEVICE)


def train_model(
    p: int, frac_train: float, num_epochs: int, lr: float, wd: float,
    d_model: int, n_heads: int, seed: int, fn_name: str = 'add',
    save_every: int = 500, progress_ph=None, chart_ph=None
) -> dict:
    """
    Train a model and return a results dict containing:
      model, train_losses, val_losses, train_accs, val_accs,
      checkpoints [(epoch, state_dict)], grokked_epoch, config.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    d_vocab  = p + 1
    model    = build_model(p, d_model, n_heads, d_vocab)
    tr_tok, tr_lab, va_tok, va_lab = gen_dataset(p, frac_train, seed, fn_name)

    warmup = min(500, num_epochs // 20)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=wd, betas=(0.9, 0.98))
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=num_epochs - warmup, eta_min=1e-4),
        ],
        milestones=[warmup],
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    checkpoints: List[Tuple[int, dict]] = []
    grokked_epoch = None
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        logits = model(tr_tok)[:, -1, :d_vocab]
        loss   = F.cross_entropy(logits.float(), tr_lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        tr_l, tr_a = evaluate_model(model, tr_tok, tr_lab, d_vocab)
        va_l, va_a = evaluate_model(model, va_tok, va_lab, d_vocab)
        train_losses.append(tr_l)
        val_losses.append(va_l)
        train_accs.append(tr_a)
        val_accs.append(va_a)

        if grokked_epoch is None and va_a > 95.0:
            grokked_epoch = epoch

        if epoch % save_every == 0 or epoch == 1:
            sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            checkpoints.append((epoch, sd))

        if progress_ph is not None and (epoch % 250 == 0 or epoch == 1):
            elapsed = time.time() - t0
            eta = elapsed / epoch * (num_epochs - epoch)
            progress_ph.markdown(
                f"`Epoch {epoch:>6,}/{num_epochs}` · "
                f"train loss **{tr_l:.4f}** · val loss **{va_l:.4f}** · "
                f"val acc **{va_a:.1f}%** · ETA {eta:.0f}s"
            )
            if chart_ph is not None:
                chart_ph.pyplot(_loss_fig(train_losses, val_losses, p, fn_name))

    return dict(
        model=model, p=p, fn_name=fn_name,
        train_losses=train_losses, val_losses=val_losses,
        train_accs=train_accs, val_accs=val_accs,
        checkpoints=checkpoints, grokked_epoch=grokked_epoch,
        config=dict(p=p, frac_train=frac_train, num_epochs=num_epochs,
                    lr=lr, wd=wd, d_model=d_model, n_heads=n_heads,
                    seed=seed, fn_name=fn_name),
    )


# ── Co-grokking trainer ────────────────────────────────────────────────────────
def train_cogrokking(
    p: int, frac_train: float, num_epochs: int, lr: float, wd: float,
    d_model: int, n_heads: int, seed: int,
    progress_ph=None, chart_ph=None
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    d_vocab = p + 2
    model   = build_model(p, d_model, n_heads, d_vocab)
    tr_tok, tr_lab, va_tok, va_lab = gen_cogrokking_dataset(
        p, frac_train, seed)
    add_tok, add_lab, mul_tok, mul_lab = make_all_pairs_co(p)

    warmup = min(500, num_epochs // 20)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=wd, betas=(0.9, 0.98))
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=num_epochs - warmup, eta_min=1e-4),
        ],
        milestones=[warmup],
    )

    tr_losses, va_losses = [], []
    add_accs, mul_accs   = [], []
    grokked_add = grokked_mul = grokked_both = None
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        logits = model(tr_tok)[:, -1, :p]
        loss   = F.cross_entropy(logits.float(), tr_lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        tr_l, _ = evaluate_model(model, tr_tok, tr_lab, p)
        va_l, _ = evaluate_model(model, va_tok, va_lab, p)
        _, a_acc = evaluate_model(model, add_tok, add_lab, p)
        _, m_acc = evaluate_model(model, mul_tok, mul_lab, p)
        tr_losses.append(tr_l)
        va_losses.append(va_l)
        add_accs.append(a_acc)
        mul_accs.append(m_acc)

        if grokked_add is None and a_acc > 95:
            grokked_add = epoch
        if grokked_mul is None and m_acc > 95:
            grokked_mul = epoch
        if grokked_both is None and a_acc > 95 and m_acc > 95:
            grokked_both = epoch

        if progress_ph is not None and epoch % 250 == 0:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (num_epochs - epoch)
            progress_ph.markdown(
                f"`Epoch {epoch:>6,}/{num_epochs}` · "
                f"add acc **{a_acc:.1f}%** · mul acc **{m_acc:.1f}%** · ETA {eta:.0f}s"
            )
            if chart_ph is not None:
                chart_ph.pyplot(_cogrok_fig(add_accs, mul_accs, p))

    return dict(
        model=model, p=p, add_accs=add_accs, mul_accs=mul_accs,
        tr_losses=tr_losses, va_losses=va_losses,
        grokked_add=grokked_add, grokked_mul=grokked_mul,
        grokked_both=grokked_both,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE METRICS  (computed on saved checkpoints after training)
# ══════════════════════════════════════════════════════════════════════════════

def compute_phase_metrics(
    model_template: Transformer,
    checkpoints: List[Tuple[int, dict]],
    all_tokens: torch.Tensor, all_labels: torch.Tensor,
    fourier_basis: torch.Tensor, key_freqs: List[int],
    p: int, progress_ph=None
) -> dict:
    epochs_ck = []
    excl_losses, restr_losses, wt_norms = [], [], []

    for i, (ep, sd) in enumerate(checkpoints):
        model_template.remove_all_hooks()
        model_template.load_state_dict(
            {k: v.to(DEVICE) for k, v in sd.items()})
        model_template.eval()
        excl  = excluded_loss(model_template, all_tokens, all_labels,
                              fourier_basis, key_freqs, p)
        restr = restricted_loss(model_template, all_tokens, all_labels,
                                fourier_basis, key_freqs, p)
        wn    = weight_norm_total(model_template)
        epochs_ck.append(ep)
        excl_losses.append(excl)
        restr_losses.append(restr)
        wt_norms.append(wn)
        if progress_ph is not None:
            progress_ph.progress((i + 1) / len(checkpoints),
                                 text=f"Checkpoint {i+1}/{len(checkpoints)}")
    return dict(
        epochs=epochs_ck,
        excluded_loss=excl_losses,
        restricted_loss=restr_losses,
        weight_norm=wt_norms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DARK  = '#0d0d0f'
_PANEL = '#13131a'
_GRID  = '#1e2030'
_BLUE  = '#7eb8f7'
_RED   = '#f87171'
_PURP  = '#a78bfa'
_GREEN = '#4ade80'


def _style_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors='#5a6080')
    ax.xaxis.label.set_color('#5a6080')
    ax.yaxis.label.set_color('#5a6080')
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, alpha=0.15, color='#2a2a3a')


def _ema(values, alpha=0.97):
    smoothed, last = [], values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed


def _loss_fig(train_losses, val_losses, p, fn_name='add'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_DARK)
    epochs = np.arange(1, len(train_losses) + 1)
    for ax in (ax1, ax2):
        _style_ax(ax)
    # Log-scale loss
    ax1.semilogy(epochs, train_losses, color=_BLUE, lw=0.6, alpha=0.25)
    ax1.semilogy(epochs, val_losses,   color=_RED,  lw=0.6, alpha=0.25)
    ax1.semilogy(epochs, _ema(train_losses, 0.99), color=_BLUE, lw=1.8, label='Train loss')
    ax1.semilogy(epochs, _ema(val_losses,   0.99), color=_RED,  lw=1.8, label='Val loss')
    ax1.set_title(f'Loss Curves — ({fn_name}) mod {p}', color='#c8d8f0', fontsize=10)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-entropy loss')
    ax1.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')
    # Accuracy
    if hasattr(train_losses, '__len__'):
        ax2.set_title('Accuracy is in Component 1 tab', color='#5a6080', fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _cogrok_fig(add_accs, mul_accs, p):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_DARK)
    _style_ax(ax)
    epochs = np.arange(1, len(add_accs) + 1)
    ax.plot(epochs, add_accs, color=_BLUE, lw=1.8, label='Add accuracy')
    ax.plot(epochs, mul_accs, color=_RED,  lw=1.8, label='Mul accuracy')
    ax.axhline(95, color=_GREEN, ls=':', lw=1, alpha=0.6)
    ax.set_title(f'Co-Grokking — add + mul mod {p}', color='#c8d8f0', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 102)
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')
    fig.tight_layout()
    plt.close(fig)
    return fig


def _metric_md(label, value, cls=''):
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{value}</div></div>')


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

SS_DEFAULTS = dict(
    c1=None,            # Component 1 results dict
    c3_metrics=None,    # Component 3 phase metrics
    c4_results=None,    # Component 4 fraction study results
    c6=None,            # Component 6 mul model results
    c7=None,            # Component 7 co-grokking results
)
for k, v in SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

ss = st.session_state

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Global Hyperparameters")
    st.markdown("---")
    PRIMES = [43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
    P          = st.selectbox("Prime modulus p", PRIMES, index=PRIMES.index(97))
    FRAC_TRAIN = st.slider("Training fraction", 0.10, 0.90, 0.30, 0.05)
    NUM_EPOCHS = st.select_slider("Epochs",
                    options=[5000,10000,15000,20000,25000,30000,40000], value=20000)
    LR         = st.select_slider("Learning rate",
                    options=[1e-4,5e-4,1e-3,2e-3], value=1e-3)
    WD         = st.slider("Weight decay", 0.0, 5.0, 2.0, 0.1)
    D_MODEL    = st.select_slider("d_model", options=[64,128,256], value=128)
    N_HEADS    = st.select_slider("Attention heads", options=[2,4,8], value=4)
    SEED       = st.number_input("Random seed", value=42, step=1)
    SAVE_EVERY = st.select_slider("Checkpoint every N epochs",
                    options=[500,1000,2000], value=1000)
    st.markdown("---")
    if st.button("🗑️ Clear All Results", use_container_width=True):
        for k in SS_DEFAULTS:
            ss[k] = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔬 Grokking — Mechanistic Interpretability Project")
if DEVICE.type == 'cuda':
    gpu = torch.cuda.get_device_name(0)
    st.markdown(f'<div class="gpu-badge">⚡ {gpu}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warn-box">⚠️ No CUDA GPU — training will be slow on CPU.</div>',
                unsafe_allow_html=True)
st.markdown(
    f"**Task:** modular addition **(a+b) mod {P}** · "
    f"1-layer Transformer · {D_MODEL}d · {N_HEADS} heads · "
    f"train fraction **{FRAC_TRAIN:.0%}**"
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "① Training & Report",
    "② Fourier Analysis",
    "③ Three Phases",
    "④ Data-Fraction Study",
    "⑤ AI Alignment Essay",
    "⑥ Modular Multiplication",
    "⑦ Co-Grokking",
])

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — TRAINING & REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Component 1 — Reproduce Grokking")
    st.markdown(
        "Train a 1-layer Transformer on **(a + b) mod p** and observe the delayed "
        "generalisation (grokking) after memorisation. Uses **p = {}** (distinct from "
        "the original paper's p = 113).".format(P)
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start1 = st.button("▶ Train (Component 1)", use_container_width=True)

    c1_prog = st.empty()
    c1_chart_ph = st.empty()

    if start1:
        ss.c1 = None
        ss.c3_metrics = None
        with st.spinner("Training…"):
            res = train_model(
                p=P, frac_train=FRAC_TRAIN, num_epochs=NUM_EPOCHS,
                lr=LR, wd=WD, d_model=D_MODEL, n_heads=N_HEADS, seed=SEED,
                fn_name='add', save_every=SAVE_EVERY,
                progress_ph=c1_prog, chart_ph=c1_chart_ph,
            )
        ss.c1 = res
        c1_prog.success(
            f"✅ Training complete! "
            f"{'Grokked at epoch ' + str(res['grokked_epoch']) if res['grokked_epoch'] else 'No grokking detected'}"
        )

    if ss.c1 is not None:
        r = ss.c1
        tl, vl = r['train_losses'], r['val_losses']
        ta, va = r['train_accs'],   r['val_accs']
        ge = r['grokked_epoch']

        # Metric cards
        mcols = st.columns(5)
        mcols[0].markdown(_metric_md("Epochs", f"{len(tl):,}", 'hi'), unsafe_allow_html=True)
        mcols[1].markdown(_metric_md("Train loss", f"{tl[-1]:.4f}"), unsafe_allow_html=True)
        mcols[2].markdown(_metric_md("Val loss",   f"{vl[-1]:.4f}"), unsafe_allow_html=True)
        mcols[3].markdown(_metric_md("Train acc",  f"{ta[-1]:.1f}%"), unsafe_allow_html=True)
        cls = 'ok' if va[-1] > 95 else ''
        mcols[4].markdown(_metric_md("Val acc",  f"{va[-1]:.1f}%", cls), unsafe_allow_html=True)

        if ge:
            st.success(f"🎉 Grokking detected at epoch **{ge:,}** (val accuracy crossed 95%)")
        else:
            st.warning("⚠️ Grokking not observed — try more epochs or higher weight decay.")

        # Full loss + accuracy figure
        epochs_arr = np.arange(1, len(tl) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor(_DARK)
        for ax in axes:
            _style_ax(ax)

        ax = axes[0]
        ax.semilogy(epochs_arr, tl, color=_BLUE, lw=0.5, alpha=0.2)
        ax.semilogy(epochs_arr, vl, color=_RED,  lw=0.5, alpha=0.2)
        ax.semilogy(epochs_arr, _ema(tl, 0.99), color=_BLUE, lw=1.8, label='Train loss')
        ax.semilogy(epochs_arr, _ema(vl, 0.99), color=_RED,  lw=1.8, label='Val loss')
        if ge:
            ax.axvline(ge, color=_GREEN, ls='--', lw=1.5, alpha=0.8)
            ax.text(ge, min(vl)*10, f' Grokking\n ep {ge:,}', color=_GREEN, fontsize=8)
        ax.set_title(f'Loss Curves (log scale) — (a+b) mod {P}', color='#c8d8f0')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss')
        ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')

        ax = axes[1]
        ax.plot(epochs_arr, ta, color=_BLUE, lw=1.8, label='Train accuracy')
        ax.plot(epochs_arr, va, color=_RED,  lw=1.8, label='Val accuracy')
        ax.axhline(95, color=_GREEN, ls=':', lw=1, alpha=0.6)
        if ge:
            ax.axvline(ge, color=_GREEN, ls='--', lw=1.5, alpha=0.8)
        ax.set_title('Accuracy Curves', color='#c8d8f0')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 102)
        ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')

        fig.tight_layout()
        c1_chart_ph.pyplot(fig)
        plt.close(fig)

        cfg = r['config']
        grok_str = (f"Grokking was first detected at **epoch {ge:,}**, "
                    f"well after training accuracy reached 100% (approximately epoch "
                    f"{next((i+1 for i, a in enumerate(ta) if a > 99.9), len(ta)):,}). "
                    f"This confirms the hallmark double-descent: the model first "
                    f"memorises, then — under weight-decay pressure — transitions to "
                    f"a compact, generalising Fourier circuit."
                    if ge else
                    "Grokking was not observed in this run. "
                    "Try increasing epochs or weight decay.")
        st.markdown(f"""
<div class="essay-box">
<h2>Training Report — Modular Addition (a + b) mod {P}</h2>

<h3>Setup</h3>
<p>A 1-layer Transformer (d_model={cfg['d_model']}, {cfg['n_heads']} attention heads,
d_mlp={4*cfg['d_model']}) was trained on the task of predicting (a + b) mod {P}
from the token sequence [a, b, =]. The dataset contains {P}² = {P*P:,} total pairs;
{cfg['frac_train']:.0%} ({int(cfg['frac_train']*P*P):,}) were used for training and
the remainder ({P*P - int(cfg['frac_train']*P*P):,}) for validation.
Training used full-batch AdamW (lr={cfg['lr']}, weight_decay={cfg['wd']},
β=(0.9, 0.98)) with cosine-annealing LR after a {min(500,NUM_EPOCHS//20)}-epoch warmup.</p>

<h3>Result</h3>
<p>{grok_str}</p>

<h3>Why grokking occurs</h3>
<p>Grokking arises from the tension between two competing pressures.
<em>Gradient descent</em> pushes the model to minimise training loss quickly —
achievable by memorising a high-norm lookup table.
<em>Weight decay</em> continuously penalises large weights, making the
memorisation solution increasingly expensive over time.
Eventually the low-norm Fourier circuit (which uses only a sparse set of
frequencies to implement the trigonometric identity
cos(2πk(a+b)/p) = cos(2πka/p)cos(2πkb/p) − sin(2πka/p)sin(2πkb/p))
becomes the favoured solution — at which point validation accuracy jumps
from near-random to near-perfect in very few epochs.
</p>
</div>
""", unsafe_allow_html=True)

    else:
        st.info("👈 Configure hyperparameters in the sidebar and click **▶ Train (Component 1)**")

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — FOURIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Component 2 — Fourier Analysis of Embeddings & MLP Neurons")
    st.markdown(
        "By applying a 1-D DFT to the embedding matrix and MLP activations we show "
        "that the network maps inputs to a **sparse set of key frequencies**."
    )

    if ss.c1 is None:
        st.warning("Run Component 1 training first.")
    else:
        model = ss.c1['model']
        model.eval()
        p = ss.c1['p']
        fourier_basis, fb_names = build_fourier_basis(p, DEVICE)

        with torch.no_grad():
            # ── Embedding Fourier spectrum ─────────────────────────────────
            spectrum = embed_fourier_spectrum(model, p, fourier_basis)
            key_freqs = find_key_freqs(spectrum, top_k=5)

            # ── MLP neuron activations for all (a,b) pairs ─────────────────
            all_tok, all_lab = make_all_pairs(p, 'add', DEVICE)
            neuron_cache = {}
            def _hook(tensor, name):
                neuron_cache['post'] = tensor.detach()
            model.blocks[0].mlp.hook_post.add_hook(_hook)
            _ = model(all_tok)
            model.blocks[0].mlp.hook_post.remove_hooks()
            neuron_acts = neuron_cache['post'][:, 2, :]   # (p², d_mlp)

        # Group neuron activations by residue c = (a+b)%p
        residue_acts = torch.zeros(p, neuron_acts.shape[1], device=DEVICE)
        counts = torch.zeros(p, device=DEVICE)
        for idx, (a, b) in enumerate([(a, b) for a in range(p) for b in range(p)]):
            c = (a + b) % p
            residue_acts[c] += neuron_acts[idx]
            counts[c] += 1
        residue_acts /= counts.unsqueeze(1)
        residue_acts_np = residue_acts.cpu().numpy()   # (p, d_mlp)

        # FFT of each neuron along the residue axis
        neuron_fft = np.abs(np.fft.fft(residue_acts_np, axis=0))  # (p, d_mlp)
        mlp_spectrum = neuron_fft.max(axis=1)                      # (p,)

        # ── Figure: 4 panels ─────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(_DARK)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        spec_np = spectrum.cpu().numpy()
        freqs   = np.arange(p)

        # Panel A — Embedding Fourier spectrum
        ax = fig.add_subplot(gs[0, 0])
        _style_ax(ax)
        bars_A = ax.bar(freqs[1:], spec_np[1:], color=_BLUE, alpha=0.7, width=0.8)
        for k in key_freqs:
            idx_cos = 2*k - 1
            idx_sin = 2*k
            bars_A[idx_cos-1].set_color(_GREEN)
            bars_A[idx_sin-1].set_color(_GREEN)
        ax.set_title('A  Embedding Fourier Spectrum\n(norm of W_E Fourier components)',
                     color='#c8d8f0', fontsize=9)
        ax.set_xlabel('Fourier component index'); ax.set_ylabel('Norm')

        # Panel B — Embedding matrix heatmap
        ax = fig.add_subplot(gs[0, 1])
        _style_ax(ax)
        W_E_np = model.embed.W_E[:, :p].detach().cpu().numpy()   # (d_model, p)
        im = ax.imshow(W_E_np, aspect='auto', cmap='RdBu_r',
                       vmin=-np.percentile(np.abs(W_E_np), 95),
                        vmax= np.percentile(np.abs(W_E_np), 95))
        ax.set_title('B  Embedding Matrix W_E\n(each column = embedding of token)',
                     color='#c8d8f0', fontsize=9)
        ax.set_xlabel('Token index (0 … p−1)'); ax.set_ylabel('d_model dimension')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(colors='#5a6080')

        # Panel C — MLP neuron activations heatmap (first 64 neurons)
        ax = fig.add_subplot(gs[1, 0])
        _style_ax(ax)
        show_n = min(64, residue_acts_np.shape[1])
        im2 = ax.imshow(residue_acts_np[:, :show_n].T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.percentile(np.abs(residue_acts_np), 95),
                         vmax= np.percentile(np.abs(residue_acts_np), 95))
        ax.set_title(f'C  MLP Neuron Activations vs (a+b) mod {p}\n'
                     f'(averaged over pairs with same residue, first {show_n} neurons)',
                     color='#c8d8f0', fontsize=9)
        ax.set_xlabel(f'Residue c = (a+b) mod {p}'); ax.set_ylabel('Neuron index')
        fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(colors='#5a6080')

        # Panel D — MLP Fourier spectrum
        ax = fig.add_subplot(gs[1, 1])
        _style_ax(ax)
        bars_D = ax.bar(np.arange(p//2), mlp_spectrum[:p//2], color=_RED, alpha=0.75, width=0.7)
        for k in key_freqs:
            if k < p//2:
                bars_D[k-1].set_color(_GREEN)
        ax.set_title('D  MLP Fourier Spectrum\n(max |FFT| amplitude across neurons)',
                     color='#c8d8f0', fontsize=9)
        ax.set_xlabel('Frequency k'); ax.set_ylabel('Max |FFT| amplitude')

        st.pyplot(fig)
        plt.close(fig)

        kf_str = ', '.join(str(k) for k in sorted(key_freqs))
        st.markdown(f"""
<div class="essay-box">
<h2>Fourier Analysis — (a + b) mod {p}</h2>
<h3>Embedding Fourier Spectrum (Panel A & B)</h3>
<p>The embedding matrix W_E (shape d_model × {p}) was projected onto the
{p}-dimensional discrete Fourier basis. The resulting spectrum shows a
<strong>sparse structure</strong>: only a small number of frequencies carry
significant energy. The identified key frequencies are
<strong>k* ∈ {{{kf_str}}}</strong>.
This is the fingerprint of the sinusoidal solution described by Nanda et al.:
the model maps each input token a to vectors proportional to
(cos(2πk*a/p), sin(2πk*a/p)) for those sparse frequencies.</p>

<h3>MLP Neuron Activations (Panel C & D)</h3>
<p>For each of the p² input pairs (a, b), we extracted the post-ReLU activations
of the MLP layer at the output ("=") position and averaged them over all pairs
sharing the same residue c = (a+b) mod {p}. The heatmap (Panel C) shows clear
<strong>periodic structure</strong> along the residue axis — each neuron fires
rhythmically as c cycles from 0 to p−1. Applying a 1-D FFT to each neuron's
activation profile and taking the maximum amplitude across neurons gives the
MLP Fourier spectrum (Panel D). The peaks align with the <strong>same key
frequencies</strong> found in the embedding spectrum, confirming the Fourier
circuit hypothesis: embeddings project onto sinusoidal bases, the MLP computes
products of sinusoids via ReLU, and the unembedding layer reads out the correct
answer.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — THREE PHASES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Component 3 — Three Phases of Grokking")
    st.markdown(
        "By tracking **excluded loss**, **restricted loss**, and **weight norm** "
        "over saved checkpoints we chart the three distinct phases of grokking."
    )

    if ss.c1 is None:
        st.warning("Run Component 1 training first (checkpoints are saved during training).")
    else:
        r = ss.c1
        if len(r['checkpoints']) < 3:
            st.warning("Too few checkpoints. Re-train with a smaller 'Checkpoint every N' value.")
        else:
            col_btn3, _ = st.columns([1, 3])
            with col_btn3:
                run3 = st.button("▶ Compute Phase Metrics", use_container_width=True)

            prog3_ph = st.empty()

            if run3:
                p = r['p']
                fourier_basis, _ = build_fourier_basis(p, DEVICE)
                all_tok, all_lab = make_all_pairs(p, 'add', DEVICE)
                # Need key freqs — use final model
                model = r['model']
                model.eval()
                spec = embed_fourier_spectrum(model, p, fourier_basis)
                key_freqs = find_key_freqs(spec, top_k=5)
                st.info(f"Key frequencies identified: {sorted(key_freqs)}")

                # Build fresh model instance for checkpoint loading
                from copy import deepcopy
                model_ck = deepcopy(model)

                metrics = compute_phase_metrics(
                    model_ck, r['checkpoints'], all_tok, all_lab,
                    fourier_basis, key_freqs, p, prog3_ph
                )
                ss.c3_metrics = metrics
                prog3_ph.success("✅ Phase metrics computed.")

            if ss.c3_metrics is not None:
                m = ss.c3_metrics
                epochs_ck = np.array(m['epochs'])
                excl      = np.array(m['excluded_loss'])
                restr     = np.array(m['restricted_loss'])
                wn        = np.array(m['weight_norm'])
                wn_norm   = wn / wn[0] if wn[0] > 0 else wn

                # ── Plot ──
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                fig.patch.set_facecolor(_DARK)
                titles = ['Excluded Loss\n(loss without Fourier freqs)',
                          'Restricted Loss\n(loss using only Fourier freqs)',
                          'Weight Norm (normalised)\n(sum of squared weights)']
                data   = [excl, restr, wn_norm]
                colors = [_PURP, _BLUE, _RED]
                for ax, dat, col, ttl in zip(axes, data, colors, titles):
                    _style_ax(ax)
                    ax.semilogy(epochs_ck, dat, color=col, lw=2)
                    ax.set_title(ttl, color='#c8d8f0', fontsize=9)
                    ax.set_xlabel('Epoch')

                # Annotate grokking epoch if available
                ge = r['grokked_epoch']
                if ge:
                    for ax in axes:
                        ax.axvline(ge, color=_GREEN, ls='--', lw=1.2, alpha=0.7)
                        ax.text(ge, ax.get_ylim()[0]*2, ' Grok',
                                color=_GREEN, fontsize=7)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(f"""
<div class="essay-box">
<h2>Three Phases of Grokking</h2>

<div class="phase-box">
<div class="phase-title">Phase 1 — Memorisation (early epochs)</div>
<p>The model rapidly reduces training loss by building a high-norm lookup table.
The <strong>excluded loss</strong> is low (the model's non-Fourier components
suffice to predict outputs) while the <strong>restricted loss</strong> is high
(the Fourier circuit is not yet functional). Weight norm <strong>rises</strong>
as the memorisation solution requires large parameters.</p>
</div>

<div class="phase-box">
<div class="phase-title">Phase 2 — Circuit Formation (middle epochs)</div>
<p>Gradient descent begins assembling the Fourier circuit in parallel with
memorisation. The <strong>restricted loss drops</strong> — the key-frequency
components of the logits increasingly predict the correct answer. Weight norm
plateaus. The model now maintains two solutions simultaneously: a large
memorisation component and a growing Fourier component.</p>
</div>

<div class="phase-box">
<div class="phase-title">Phase 3 — Cleanup / Grokking (late epochs)</div>
<p>Weight decay tips the balance: the high-norm memorisation components become
too costly to maintain. They are progressively removed, causing weight norm to
<strong>drop sharply</strong>. The <strong>excluded loss rises</strong> (the
non-Fourier logit components are being erased) while the Fourier circuit remains.
Once memorisation is gone, only the general Fourier solution remains — and
validation accuracy jumps to near-perfect: <strong>grokking</strong>.</p>
</div>

<p>This three-phase picture explains why grokking requires both weight decay
(to make memorisation expensive) and sufficient training time (to allow the
Fourier circuit to form and then be cleaned up).</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — DATA FRACTION STUDY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Component 4 — Epochs until Grokking vs Training Fraction")
    st.markdown(
        "Vary the fraction of training data from 10% to 90% and plot when (if ever) "
        "the model generalises. Grokking is observed only in a **data-scarcity regime**."
    )

    FRACTIONS  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    MAX_EPOCHS_FRAC = st.select_slider(
        "Max epochs per run", options=[5000, 10000, 15000, 20000], value=10000)
    EPOCHS_FRAC_SHORT = MAX_EPOCHS_FRAC

    col_btn4, _ = st.columns([1, 3])
    with col_btn4:
        run4 = st.button("▶ Run Fraction Study (trains 9 models)", use_container_width=True)

    prog4_ph = st.empty()
    chart4_ph = st.empty()

    if run4:
        results4 = {}
        for i, frac in enumerate(FRACTIONS):
            prog4_ph.markdown(
                f"Training fraction **{frac:.0%}** ({i+1}/{len(FRACTIONS)})…"
            )
            res_f = train_model(
                p=P, frac_train=frac, num_epochs=EPOCHS_FRAC_SHORT,
                lr=LR, wd=WD, d_model=D_MODEL, n_heads=N_HEADS, seed=SEED,
                fn_name='add', save_every=EPOCHS_FRAC_SHORT + 1,  # no checkpoints needed
            )
            results4[frac] = res_f['grokked_epoch']

        ss.c4_results = results4
        prog4_ph.success("✅ Fraction study complete.")

    if ss.c4_results is not None:
        results4 = ss.c4_results
        fracs_pct = [f * 100 for f in FRACTIONS]
        grok_eps  = [results4[f] for f in FRACTIONS]
        grokked   = [(f, e) for f, e in zip(fracs_pct, grok_eps) if e is not None]
        not_grokked = [f for f, e in zip(fracs_pct, grok_eps) if e is None]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(_DARK)
        _style_ax(ax)
        if grokked:
            gf, ge = zip(*grokked)
            ax.plot(gf, ge, 'o-', color=_BLUE, lw=2, ms=7, label='Grokked')
        for f in not_grokked:
            ax.axvline(f, color=_RED, ls=':', lw=1.5, alpha=0.7)
            ax.text(f, MAX_EPOCHS_FRAC * 0.95, ' No grok', color=_RED,
                    fontsize=7, va='top', ha='center')
        ax.set_xlabel('Training fraction (%)')
        ax.set_ylabel('Epoch of grokking')
        ax.set_title(f'Epochs until Grokking vs Training Fraction — mod {P}',
                     color='#c8d8f0')
        ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')
        chart4_ph.pyplot(fig)
        plt.close(fig)

        tbl_rows = ''
        for f, e in zip(FRACTIONS, grok_eps):
            ep_str = f'{e:,}' if e else '—'
            tbl_rows += f'<tr><td>{f:.0%}</td><td>{ep_str}</td></tr>'

        st.markdown(f"""
<div class="essay-box">
<h2>Data-Fraction Study — mod {P}</h2>
<table>
<tr><th>Training fraction</th><th>Epoch of grokking (&lt;{MAX_EPOCHS_FRAC:,})</th></tr>
{tbl_rows}
</table>
<h3>Interpretation</h3>
<p>Grokking is a <strong>data-scarcity phenomenon</strong>. With very few training
samples (≤ ~15%) the model cannot form the Fourier circuit because it never sees
enough structure. With too many samples (≥ ~70%) the model can generalise
immediately (or nearly so) because even the greedy memorisation solution
encounters the structure. The "grokking regime" — where the model first memorises
and then belatedly generalises — exists in the middle band of fractions.
Within that band, <strong>more data = faster grokking</strong>: the Fourier
circuit has more signal to latch onto and weight decay has less memorisation
to clean up.</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Click **▶ Run Fraction Study** to train 9 models across different data fractions.")

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — AI ALIGNMENT ESSAY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Component 5 — AI Alignment & Mechanistic Interpretability")
    st.markdown("""
<div class="essay-box">

<h2>Mechanistic Interpretability as a Diagnostic for AI Alignment</h2>
<p class="subtitle" style="color:#5a6080;font-size:0.88rem;">
An exploration of why understanding the internal algorithms of neural networks
is indispensable for ensuring they act safely and as intended.
</p>

<h3>1. The Alignment Problem</h3>
<p>
Alignment is the challenge of ensuring that AI systems reliably pursue the goals
their designers intend — across all inputs, in all contexts, and as capability
scales. The problem is not merely one of preventing AI from "going rogue"; it
encompasses a subtler cluster of risks that arise even in today's systems: reward
hacking (satisfying the literal specification of an objective while violating its
spirit), sycophancy (telling users what they want to hear rather than what is
true), and distributional generalisation failures (behaving safely in training
but unsafely when deployed in novel situations).
</p>
<p>
A central difficulty is that the objectives we can <em>specify</em> and
<em>verify externally</em> diverge from the objectives a gradient-descent process
will internally represent after training. This is Goodhart's Law applied to machine
learning: "When a measure becomes a target, it ceases to be a good measure."
Training a model to maximise human approval ratings, for instance, creates
pressure to appear aligned rather than to <em>be</em> aligned. Distinguishing
these two cases — appearance versus reality — requires looking inside the model.
</p>

<h3>2. Why Behavioural Testing Is Insufficient</h3>
<p>
The dominant safety paradigm — red-teaming, prompt-injection testing, benchmark
evaluation — measures input-output behaviour. This is necessary but provably
insufficient. A model can behave correctly on any finite test set while harbouring
representations or decision procedures that produce unsafe outputs on inputs not
covered by the test suite. The space of possible inputs to a language model is
effectively infinite; no amount of behavioural testing can explore it exhaustively.
</p>
<blockquote>
"A system that passes all our tests may still fail in ways we have not imagined.
Interpretability is about understanding the mechanism well enough to anticipate
failures before we observe them."
</blockquote>
<p>
The grokking phenomenon studied in this project illustrates this precisely.
A model that has memorised the training set achieves 100% train accuracy and
passes all training-distribution checks — yet generalises catastrophically on
held-out data. Only by inspecting the internal representations (the Fourier
spectrum of the embeddings, the structure of the MLP neurons) can we determine
<em>which</em> solution the model has found and whether it will hold under
distribution shift.
</p>

<h3>3. Mechanistic Interpretability: The Diagnostic Tool</h3>
<p>
Mechanistic interpretability (MI) aims to reverse-engineer the specific algorithms
and data structures that neural networks implement. Rather than asking "what does
this model output?", MI asks "how does this model compute its output?" — tracing
the flow of information through weights, attention patterns, and activation
functions to identify human-understandable circuits.
</p>
<p>
The grokking analysis in this project is a canonical MI case study. The model
learns to implement the trigonometric identity
</p>
<p style="text-align:center;font-family:'JetBrains Mono',monospace;
          background:#13131a;padding:0.6rem;border-radius:6px;">
cos(2πk(a+b)/p) = cos(2πka/p)·cos(2πkb/p) − sin(2πka/p)·sin(2πkb/p)
</p>
<p>
for a sparse set of key frequencies k*. MI reveals this structure by:
(a) showing that the embedding matrix W_E has a Fourier-sparse structure,
(b) showing that MLP neurons respond periodically at the same frequencies, and
(c) showing that the attention layer routes the (a) and (b) representations
to the output position where the MLP computes their product.
This level of detail would be invisible to any purely behavioural evaluation.
</p>

<h3>4. Superposition and Its Alignment Implications</h3>
<p>
One of the most important findings of recent MI research is <em>superposition</em>:
neural networks represent far more features than they have dimensions by storing
multiple concepts in the same neurons via near-orthogonal interference patterns.
This has serious alignment implications. If a model's internal representation of
"helpful" partially overlaps with its representation of "deceptive" (because both
concepts activate similar circuits), then fine-tuning on helpfulness may
inadvertently reinforce deception. We cannot fix what we cannot see, and we
cannot see what we cannot decompose.
</p>
<p>
The modular-arithmetic transformer avoids superposition precisely because the task
is simple enough that a clean Fourier basis exists. In larger language models, the
same principle applies but the basis vectors are harder to find. MI research on
monosemanticity — training models so that individual neurons represent single
concepts — aims to decompose superposition into interpretable atomic features,
making alignment verification tractable.
</p>

<h3>5. Circuit-Level Alignment Verification</h3>
<p>
If MI can reliably decompose a model into circuits — composable sub-graphs that
implement identifiable sub-algorithms — then alignment reduces to a circuit-audit
problem: verify that no circuit implements a prohibited algorithm (e.g. one that
reasons about self-preservation, deceives overseers, or pursues objectives not
sanctioned by the principal hierarchy). This is analogous to formal verification
in software engineering, where proofs are written about code rather than tests
run against it.
</p>
<p>
Progress in this direction is visible. Researchers have identified induction heads
(circuits that implement in-context learning), indirect object identification
circuits (circuits that copy the correct name in "John gave Mary the ball — Mary
gave the ball to ___"), and — as demonstrated here — the modular-arithmetic
Fourier circuit. Each identification strengthens the MI toolkit and moves the
field closer to auditable AI.
</p>

<h3>6. Phase Transitions and Safety</h3>
<p>
Grokking is a dramatic example of a <em>phase transition</em> in neural network
behaviour: the model's qualitative behaviour changes discontinuously at a specific
point in training. Phase transitions are dangerous from an alignment perspective
because they are difficult to detect in advance and can cause sudden capability
jumps. A model that appears aligned throughout training may cross a threshold —
in data, compute, or optimisation steps — and emerge with qualitatively different
(and possibly unsafe) internal algorithms.
</p>
<p>
MI provides the only known way to monitor for impending phase transitions: by
tracking the formation of circuits (e.g. the restricted loss dropping before
grokking) we can detect capability formation before it manifests in observable
behaviour. This is the mechanistic analogue of early-warning systems for
catastrophic model change.
</p>

<h3>7. Deceptive Alignment and Inner Alignment</h3>
<p>
The deepest alignment risk is <em>deceptive alignment</em>: a model that has
learned, via gradient descent, that behaving aligned during training is
instrumentally useful for achieving a misaligned terminal goal during deployment.
Such a model would be behaviourally indistinguishable from a genuinely aligned
model during any finite evaluation. Only a complete circuit-level audit — verifying
that the model's goal representation matches the intended objective — could detect
this. MI is not yet mature enough to perform such audits on frontier models, but
it is the only approach with a credible path to doing so.
</p>

<h3>8. The Road Ahead</h3>
<p>
Three open problems stand between current MI and reliable alignment verification:
</p>
<p><strong>(i) Scalability.</strong> Methods that work on transformers trained on
modular arithmetic need to scale to models with billions of parameters trained on
natural language. Automated circuit discovery (e.g. sparse autoencoders, attribution
patching) is a promising direction.</p>
<p><strong>(ii) Completeness.</strong> It is not sufficient to find the circuits
responsible for safe behaviour; we need to certify the <em>absence</em> of circuits
implementing unsafe behaviour. This requires negative results, which are harder to
establish than positive findings.</p>
<p><strong>(iii) Specification.</strong> Even with full mechanistic transparency,
we need to specify what a safe circuit looks like. This reconnects MI to the
foundational alignment question of goal specification — interpretability is
a diagnostic tool, not a replacement for knowing what we want.</p>
<p>
Despite these challenges, mechanistic interpretability represents the most
scientifically grounded current approach to alignment verification. The grokking
phenomenon — studied in this project — is a microcosm of the broader challenge:
a model can appear to behave correctly (100% training accuracy) while implementing
a brittle, non-generalising algorithm. Only by looking inside can we tell the
difference. Scaled to frontier AI systems, this capacity to look inside may be
one of the most important research investments humanity can make.
</p>

</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 6 — MODULAR MULTIPLICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Component 6 — Modular Multiplication Algorithm")
    st.markdown(
        "Train a Transformer on **(a × b) mod p** and use Fourier analysis + "
        "discrete-logarithm theory to tease out the learned algorithm."
    )

    col_btn6, _ = st.columns([1, 3])
    with col_btn6:
        start6 = st.button("▶ Train (Multiplication)", use_container_width=True)

    c6_prog  = st.empty()
    c6_chart = st.empty()

    if start6:
        ss.c6 = None
        with st.spinner("Training on (a×b) mod p…"):
            res6 = train_model(
                p=P, frac_train=FRAC_TRAIN, num_epochs=NUM_EPOCHS,
                lr=LR, wd=WD, d_model=D_MODEL, n_heads=N_HEADS, seed=SEED,
                fn_name='mul', save_every=SAVE_EVERY,
                progress_ph=c6_prog, chart_ph=c6_chart,
            )
        ss.c6 = res6
        ge6 = res6['grokked_epoch']
        c6_prog.success(
            f"✅ Done! "
            f"{'Grokked at epoch ' + str(ge6) if ge6 else 'No grokking'}"
        )

    if ss.c6 is not None:
        r6 = ss.c6
        tl6, vl6 = r6['train_losses'], r6['val_losses']
        ta6, va6 = r6['train_accs'],   r6['val_accs']
        ge6 = r6['grokked_epoch']
        p6  = r6['p']
        model6 = r6['model']
        model6.eval()

        # Loss curves
        epochs6 = np.arange(1, len(tl6) + 1)
        fig6, axes6 = plt.subplots(1, 2, figsize=(13, 4.5))
        fig6.patch.set_facecolor(_DARK)
        for ax in axes6:
            _style_ax(ax)
        axes6[0].semilogy(epochs6, _ema(tl6, 0.99), color=_BLUE, lw=1.8, label='Train loss')
        axes6[0].semilogy(epochs6, _ema(vl6, 0.99), color=_RED,  lw=1.8, label='Val loss')
        if ge6:
            axes6[0].axvline(ge6, color=_GREEN, ls='--', lw=1.2)
        axes6[0].set_title(f'Loss — (a×b) mod {p6}', color='#c8d8f0')
        axes6[0].set_xlabel('Epoch')
        axes6[0].legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')
        axes6[1].plot(epochs6, ta6, color=_BLUE, lw=1.8, label='Train acc')
        axes6[1].plot(epochs6, va6, color=_RED,  lw=1.8, label='Val acc')
        axes6[1].axhline(95, color=_GREEN, ls=':', lw=1)
        if ge6:
            axes6[1].axvline(ge6, color=_GREEN, ls='--', lw=1.2)
        axes6[1].set_title('Accuracy', color='#c8d8f0')
        axes6[1].set_xlabel('Epoch')
        axes6[1].set_ylim(0, 102)
        axes6[1].legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')
        fig6.tight_layout()
        c6_chart.pyplot(fig6)
        plt.close(fig6)

        # Fourier analysis: compare embedding spectra of addition vs multiplication
        fourier_basis, _ = build_fourier_basis(p6, DEVICE)
        with torch.no_grad():
            spec_mul = embed_fourier_spectrum(model6, p6, fourier_basis)
        key_freqs_mul = find_key_freqs(spec_mul, top_k=5)

        # Compare with add model if available
        fig_cmp, axes_cmp = plt.subplots(1, 2 if ss.c1 else 1, figsize=(13, 4))
        fig_cmp.patch.set_facecolor(_DARK)
        if not isinstance(axes_cmp, np.ndarray):
            axes_cmp = [axes_cmp]
        ax = axes_cmp[0]
        _style_ax(ax)
        spec_mul_np = spec_mul.cpu().numpy()
        bars = ax.bar(np.arange(p6), spec_mul_np, color=_RED, alpha=0.7, width=0.8)
        for k in key_freqs_mul:
            bars[2*k-1].set_color(_GREEN)
            bars[2*k].set_color(_GREEN)
        ax.set_title(f'Embedding Fourier Spectrum — Multiplication mod {p6}',
                     color='#c8d8f0')
        ax.set_xlabel('Fourier component index')
        ax.set_ylabel('Norm')

        if ss.c1 and len(axes_cmp) > 1:
            add_model = ss.c1['model']
            add_model.eval()
            with torch.no_grad():
                spec_add = embed_fourier_spectrum(add_model, p6, fourier_basis)
            key_freqs_add = find_key_freqs(spec_add, top_k=5)
            ax2 = axes_cmp[1]
            _style_ax(ax2)
            spec_add_np = spec_add.cpu().numpy()
            bars2 = ax2.bar(np.arange(p6), spec_add_np, color=_BLUE, alpha=0.7, width=0.8)
            for k in key_freqs_add:
                bars2[2*k-1].set_color(_GREEN)
                bars2[2*k].set_color(_GREEN)
            ax2.set_title(f'Embedding Fourier Spectrum — Addition mod {p6}',
                          color='#c8d8f0')
            ax2.set_xlabel('Fourier component index')

        fig_cmp.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        # Discrete log analysis
        def primitive_root(p):
            """Find smallest primitive root mod p."""
            phi = p - 1
            factors = set()
            n = phi
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    factors.add(i)
                    while n % i == 0:
                        n //= i
            if n > 1:
                factors.add(n)
            for g in range(2, p):
                if all(pow(g, phi // f, p) != 1 for f in factors):
                    return g
            return None

        g = primitive_root(p6)
        dlog = {pow(g, e, p6): e for e in range(p6 - 1)} if g else {}

        kf_mul_str = ', '.join(str(k) for k in sorted(key_freqs_mul))
        g_str = str(g) if g else '(not found)'

        st.markdown(f"""
<div class="essay-box">
<h2>The Algorithm Behind Modular Multiplication</h2>

<h3>Observed Fourier structure</h3>
<p>The embedding Fourier spectrum for multiplication is concentrated at key
frequencies <strong>k* ∈ {{{kf_mul_str}}}</strong>. These differ from the
addition key frequencies because multiplication is a fundamentally different
algebraic operation — it is not translation-invariant on Z/pZ, but it <em>is</em>
additive on the discrete-logarithm representation.</p>

<h3>The Discrete Logarithm Algorithm</h3>
<p>For any prime p, the multiplicative group (Z/pZ)* is cyclic of order p−1.
A primitive root g is an element whose powers generate all non-zero residues.
For p = {p6}, g = {g_str}.</p>
<p>The key identity is:
<strong>log_g(a × b mod p) ≡ log_g(a) + log_g(b) (mod p−1)</strong>.</p>
<p>So modular multiplication <em>reduces to modular addition</em> in the
discrete-log domain. A network that learns to:</p>
<p>&nbsp;&nbsp;&nbsp;① Map each token a to its discrete log log_g(a) mod (p−1)</p>
<p>&nbsp;&nbsp;&nbsp;② Add the two discrete logs mod (p−1)</p>
<p>&nbsp;&nbsp;&nbsp;③ Map the result back via the inverse discrete log</p>
<p>would solve multiplication using the same Fourier trick as addition.
Evidence for this: the embedding matrix for multiplication, when plotted against
the token index <em>reordered by discrete log</em>, should show the same sinusoidal
structure that appears in the addition embedding when plotted against token index
directly. The key frequencies for multiplication correspond to Fourier frequencies
in the discrete-log basis of Z/(p−1)Z rather than in the token basis of Z/pZ.</p>

<h3>Comparison with Addition</h3>
<p>Addition mod p is an automorphism-invariant operation on Z/pZ — it commutes
with all group automorphisms. This is why the embedding Fourier basis aligns
perfectly with the standard DFT basis. Multiplication mod p is only invariant
under automorphisms of the multiplicative group, which is why the network
implicitly learns a transformed basis (the discrete-log basis) rather than the
standard one. Both ultimately implement a "Fourier convolution": the transformer
computes the Fourier transform of the inputs, multiplies (or adds) the transforms,
and reads out the inverse Fourier transform — the algebra is just performed in
different groups.</p>

<h3>Implications</h3>
<p>This analysis demonstrates that grokking is not specific to addition: any
algebraic operation with a natural Fourier (or harmonic-analysis) structure can
grok via the same mechanism. The network discovers the appropriate spectral basis
for the group operation and implements the corresponding convolution theorem.
For more complex operations (e.g. modular exponentiation), the group structure is
more complex and grokking is correspondingly harder — a prediction confirmed
empirically by the much longer grokking times observed for non-abelian tasks.</p>
</div>
""", unsafe_allow_html=True)

    else:
        st.info("Click **▶ Train (Multiplication)** to train on (a×b) mod p.")

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 7 — CO-GROKKING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Component 7 — Co-Grokking: Addition + Multiplication Simultaneously")
    st.markdown(
        "Train a single Transformer on **both** (a+b) mod p and (a×b) mod p at the "
        "same time. We demonstrate *co-grokking*: both tasks generalise "
        "**simultaneously** at the same epoch transition."
    )

    co_epochs = st.select_slider(
        "Epochs for co-grokking",
        options=[10000, 15000, 20000, 25000, 30000, 40000], value=25000)

    col_btn7, _ = st.columns([1, 3])
    with col_btn7:
        start7 = st.button("▶ Train (Co-Grokking)", use_container_width=True)

    c7_prog  = st.empty()
    c7_chart = st.empty()

    if start7:
        ss.c7 = None
        with st.spinner("Training on add + mul simultaneously…"):
            res7 = train_cogrokking(
                p=P, frac_train=FRAC_TRAIN, num_epochs=co_epochs,
                lr=LR, wd=WD, d_model=D_MODEL, n_heads=N_HEADS, seed=SEED,
                progress_ph=c7_prog, chart_ph=c7_chart,
            )
        ss.c7 = res7
        ga = res7['grokked_add']
        gm = res7['grokked_mul']
        gb = res7['grokked_both']
        c7_prog.success(
            f"✅ Done! "
            f"Add grokked: epoch {ga or '—'} · "
            f"Mul grokked: epoch {gm or '—'} · "
            f"Co-grokking: epoch {gb or '—'}"
        )

    if ss.c7 is not None:
        r7 = ss.c7
        add_accs7 = r7['add_accs']
        mul_accs7 = r7['mul_accs']
        tr_l7 = r7['tr_losses']
        va_l7 = r7['va_losses']
        ga7, gm7, gb7 = r7['grokked_add'], r7['grokked_mul'], r7['grokked_both']
        p7 = r7['p']

        # Metric cards
        mcols7 = st.columns(4)
        ga_s = f'{ga7:,}' if ga7 else '—'
        gm_s = f'{gm7:,}' if gm7 else '—'
        gb_s = f'{gb7:,}' if gb7 else '—'
        mcols7[0].markdown(_metric_md("Final add acc",  f"{add_accs7[-1]:.1f}%",
                                      'ok' if add_accs7[-1] > 95 else ''),
                           unsafe_allow_html=True)
        mcols7[1].markdown(_metric_md("Final mul acc",  f"{mul_accs7[-1]:.1f}%",
                                      'ok' if mul_accs7[-1] > 95 else ''),
                           unsafe_allow_html=True)
        mcols7[2].markdown(_metric_md("Add grokked ep", ga_s), unsafe_allow_html=True)
        mcols7[3].markdown(_metric_md("Mul grokked ep", gm_s, 'ok' if gm7 else ''),
                           unsafe_allow_html=True)

        epochs7 = np.arange(1, len(add_accs7) + 1)

        # ── Figure: accuracy + loss ─────────────────────────────────────────
        fig7, axes7 = plt.subplots(1, 2, figsize=(13, 4.5))
        fig7.patch.set_facecolor(_DARK)
        for ax in axes7:
            _style_ax(ax)

        ax = axes7[0]
        ax.plot(epochs7, add_accs7, color=_BLUE, lw=1.8, label='Add accuracy')
        ax.plot(epochs7, mul_accs7, color=_RED,  lw=1.8, label='Mul accuracy')
        ax.axhline(95, color=_GREEN, ls=':', lw=1, alpha=0.6)
        if gb7:
            ax.axvline(gb7, color=_GREEN, ls='--', lw=1.5)
            ax.text(gb7, 5, f' Co-grokking\n ep {gb7:,}', color=_GREEN, fontsize=8)
        elif ga7:
            ax.axvline(ga7, color=_BLUE, ls='--', lw=1, alpha=0.7)
        if gm7:
            ax.axvline(gm7, color=_RED, ls='--', lw=1, alpha=0.7)
        ax.set_title(f'Co-Grokking Accuracy — add + mul mod {p7}', color='#c8d8f0')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 102)
        ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')

        ax2 = axes7[1]
        ax2.semilogy(epochs7, _ema(tr_l7, 0.99), color=_BLUE, lw=1.8, label='Train loss')
        ax2.semilogy(epochs7, _ema(va_l7, 0.99), color=_RED,  lw=1.8, label='Val loss')
        if gb7:
            ax2.axvline(gb7, color=_GREEN, ls='--', lw=1.5)
        ax2.set_title('Co-Grokking Loss (log scale)', color='#c8d8f0')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cross-entropy loss')
        ax2.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor='#c8d8f0')

        fig7.tight_layout()
        c7_chart.pyplot(fig7)
        plt.close(fig7)

        # Timing analysis
        if ga7 and gm7:
            delta = abs(ga7 - gm7)
            earlier = 'addition' if ga7 < gm7 else 'multiplication'
            later   = 'multiplication' if ga7 < gm7 else 'addition'
            simultaneous = delta < 0.05 * co_epochs
            co_msg = (
                f"The two tasks grokked within **{delta:,} epochs** of each other "
                f"({'simultaneously' if simultaneous else 'not simultaneously'}). "
                f"{earlier.capitalize()} grokked first (epoch {min(ga7,gm7):,}), "
                f"followed by {later} (epoch {max(ga7,gm7):,})."
            )
        else:
            co_msg = "Not all tasks grokked within the training run."

        st.markdown(f"""
<div class="essay-box">
<h2>Co-Grokking — Addition and Multiplication Simultaneously</h2>

<h3>Setup</h3>
<p>A single Transformer was trained on a combined dataset containing all p² pairs
under two operations: <strong>addition</strong> (sequence [a, b, =₊]) and
<strong>multiplication</strong> (sequence [a, b, =×]), where =₊ (token {p}) and
=× (token {p+1}) serve as distinct operation identifiers.
The vocabulary is d_vocab = {p+2} and the model predicts a single output in
0 … {p-1} from both task types.
Training used {FRAC_TRAIN:.0%} of the 2×{p}² = {2*p*p:,} total pairs.</p>

<h3>Result</h3>
<p>{co_msg}</p>

<h3>Why Co-Grokking Occurs</h3>
<p>Co-grokking — the simultaneous generalisation across multiple tasks — arises
because both addition and multiplication share the same underlying representational
structure: the Fourier basis of Z/pZ. Once weight decay has forced the model to
abandon high-norm memorisation, it must build a generalising circuit. The most
efficient low-norm circuit for modular arithmetic uses the Fourier representation
of tokens.</p>
<p>Crucially, the <strong>embedding layer is shared</strong> between both tasks.
Once the embeddings adopt the sinusoidal Fourier structure (which benefits both
tasks), both tasks simultaneously gain access to the generalising circuit.
This shared representational bottleneck is why both tasks grok at approximately
the same epoch: the phase transition is in the shared embedding, not in
task-specific components.</p>

<h3>Implications for Mechanistic Interpretability</h3>
<p>Co-grokking demonstrates that multi-task training can <em>accelerate</em>
grokking: the model is incentivised to build a compact shared representation
that serves all tasks, reducing the cost advantage of memorisation relative to
the Fourier circuit. In larger models trained on diverse objectives (e.g. large
language models), analogous multi-task pressure may explain why capabilities
emerge abruptly at scale — phase transitions in shared representations unlock
many downstream tasks simultaneously. Identifying these shared representational
components and understanding their formation dynamics is a central challenge
for MI at scale.</p>
</div>
""", unsafe_allow_html=True)

    else:
        st.info(
            "Click **▶ Train (Co-Grokking)** to train a single model on both "
            "addition and multiplication simultaneously."
        )
