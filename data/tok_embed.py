# data/tok_embed.py 
# Input embeddings for discrete tokens:
#   - Token embeddings (E_V), optional factorization
#   - Positional embeddings (sinusoidal por defecto, o learned)
#   - Weight tying para la cabeza de salida (logits)
# PyTorch-only, autocontenido.

from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ------------------------------- Vocab mínima --------------------------------

class SimpleVocab:
    """
    Vocabulario mínimo con especiales configurables.
    Por defecto incluye: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
    Puedes desactivar UNK (p.ej. add_unk=False si tienes vocab completo).
    """
    def __init__(
        self,
        tokens: Iterable[str],
        *,
        add_pad: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
        add_unk: bool = True,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ):
        toks = list(tokens)
        specials: List[str] = []
        if add_pad: specials.append(pad_token)
        if add_bos: specials.append(bos_token)
        if add_eos: specials.append(eos_token)
        if add_unk: specials.append(unk_token)

        # Evita duplicados
        seen = set()
        ordered = []
        for t in specials + toks:
            if t not in seen:
                seen.add(t)
                ordered.append(t)

        self.itos: List[str] = ordered
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

        self.pad_token, self.bos_token, self.eos_token, self.unk_token = pad_token, bos_token, eos_token, unk_token
        self.pad_id  = self.stoi.get(pad_token, None)
        self.bos_id  = self.stoi.get(bos_token, None)
        self.eos_id  = self.stoi.get(eos_token, None)
        self.unk_id  = self.stoi.get(unk_token, None)

        if self.pad_id is None:
            raise ValueError("SimpleVocab requiere un token <pad> (add_pad=True).")

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], mode: str = "char", **kwargs) -> "SimpleVocab":
        mode = mode.lower()
        if mode == "char":
            symbols = sorted({ch for text in texts for ch in text})
        elif mode == "word":
            symbols = sorted({w for text in texts for w in text.split()})
        else:
            raise ValueError("mode must be 'char' or 'word'")
        return cls(symbols, **kwargs)

    def encode(self, text: str, mode: str = "char", add_bos: bool = False, add_eos: bool = False) -> List[int]:
        mode = mode.lower()
        if mode == "char":
            toks = list(text)
        elif mode == "word":
            toks = text.split()
        else:
            raise ValueError("mode must be 'char' or 'word'")
        ids = [self.stoi.get(t, self.unk_id if self.unk_id is not None else self.pad_id) for t in toks]
        if add_bos and self.bos_id is not None: ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None: ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], mode: str = "char") -> str:
        toks = [self.itos[i] if 0 <= i < len(self.itos) else (self.unk_token if self.unk_id is not None else self.pad_token) for i in ids]
        return "".join(toks) if mode == "char" else " ".join(toks)

    @property
    def size(self) -> int:
        return len(self.itos)


def pack_batch(seqs: List[List[int]], pad_id: int, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """
    Pad a lista de secuencias -> (tokens[B,T], mask[B,T]); mask=1 donde hay token real.
    """
    B = len(seqs)
    T = max((len(s) for s in seqs), default=1)
    x = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    m = torch.zeros((B, T), dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        if not s: continue
        n = min(len(s), T)
        x[i, :n] = torch.tensor(s[:n], dtype=torch.long, device=device)
        m[i, :n] = True
    return x, m


# ---------------------------- Positional embeddings ---------------------------

def _sinusoidal_positional_encoding(T: int, d_model: int, device=None, dtype=torch.float32) -> Tensor:
    pe = torch.zeros(T, d_model, device=device, dtype=dtype)
    position = torch.arange(0, T, device=device, dtype=dtype).unsqueeze(1)  # [T,1]
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d]


# --------------------------------- Embedder -----------------------------------

class TokenEmbedder(nn.Module):
    """
    Token + positional embedding con factor opcional y weight tying.

    Args:
      vocab_size: V
      d_model: d
      pos_kind: 'sinusoidal' (defecto) | 'learned'
      max_len: requerido si pos_kind='learned'
      dropout: post-suma token+pos
      tie_softmax: si True, usa el mismo E_V para logits (tying)
      factor_rank: si >0, factoriza E_V = E_in[V,r] · P[r,d]
      pad_id: id de padding (para máscaras / ignore_index)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        pos_kind: str = "sinusoidal",
        max_len: Optional[int] = None,
        dropout: float = 0.0,
        tie_softmax: bool = True,
        factor_rank: Optional[int] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size y d_model deben ser > 0.")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.pad_id = int(pad_id)
        self.tie_softmax = bool(tie_softmax)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Factorización opcional
        self.factor_rank = int(factor_rank) if (factor_rank is not None and factor_rank > 0) else None
        if self.factor_rank is None:
            self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)  # E_V
            self.proj = None
        else:
            r = self.factor_rank
            self.token_emb = nn.Embedding(self.vocab_size, r, padding_idx=self.pad_id)  # E_in
            self.proj = nn.Linear(r, self.d_model, bias=False)                           # P
            nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(r))

        # Positional
        pos_kind = pos_kind.lower()
        if pos_kind not in {"sinusoidal", "learned"}:
            raise ValueError("pos_kind debe ser 'sinusoidal' o 'learned'.")
        self.pos_kind = pos_kind
        if pos_kind == "learned":
            if max_len is None or max_len <= 0:
                raise ValueError("max_len es obligatorio para pos_kind='learned'.")
            self.pos_emb = nn.Embedding(max_len, self.d_model)
        else:
            self.register_buffer("_pe_cache", torch.zeros(1, 1, self.d_model), persistent=False)  # [1,1,d]
            self._pe_len = 1

        # Bias para tying
        if self.tie_softmax:
            self.out_bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.register_parameter("out_bias", None)

        # Inits
        nn.init.normal_(self.token_emb.weight, std=1.0 / math.sqrt(self.d_model if self.factor_rank is None else self.factor_rank))
        if isinstance(getattr(self, "pos_emb", None), nn.Embedding):
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    # ------------------------------ Utils internas -----------------------------

    def _effective_token_weight(self) -> Tensor:
        """Peso efectivo W ∈ ℝ^{V×d} para tying."""
        if self.factor_rank is None or self.proj is None:
            return self.token_emb.weight
        return self.token_emb.weight @ self.proj.weight.T  # [V,d]

    def _positional_for_length(self, T: int, device, dtype) -> Tensor:
        if self.pos_kind == "learned":
            if T > self.pos_emb.num_embeddings:
                raise ValueError(f"L={T} > max_len={self.pos_emb.num_embeddings}.")
            return self.pos_emb(torch.arange(T, device=device))
        if T > self._pe_len:
            pe = _sinusoidal_positional_encoding(T, self.d_model, device=device, dtype=dtype)
            self._pe_len = T
            self._pe_cache = pe.unsqueeze(0)
        return self._pe_cache[0, :T, :]  # [T,d]

    # --------------------------------- API ------------------------------------

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids [B,T] -> h0 [B,T,d] con E_V + E_d.
        """
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids debe ser torch.long")
        B, T = token_ids.shape
        dev = token_ids.device
        tok = self.token_emb(token_ids)                  # [B,T,r|d]
        if self.proj is not None:
            tok = self.proj(tok)                         # [B,T,d]
        pos = self._positional_for_length(T, dev, tok.dtype).unsqueeze(0).expand(B, T, self.d_model)
        h = tok + pos
        return self.dropout(h)

    def logits_from_hidden(self, h: Tensor, mask: Optional[Tensor] = None, mask_behavior: str = "none") -> Tensor:
        """
        Devuelve logits con tying: logits = h @ W^T + b. Opcionalmente aplica máscara:
          - mask: [B,T] bool (1=pos válida, 0=ignorar)
          - mask_behavior:
              'none'   -> no tocar logits (usa ignore_index en CE)
              'neg_inf'-> logits[mask==0,:] = -inf  (ojo si haces softmax)
              'zero'   -> logits[mask==0,:] = 0.0   (seguro para softmax)
        """
        if not self.tie_softmax:
            raise RuntimeError("logits_from_hidden requiere tie_softmax=True.")
        W = self._effective_token_weight()              # [V,d]
        logits = torch.matmul(h, W.T)                   # [B,T,V]
        if self.out_bias is not None:
            logits = logits + self.out_bias
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            m = ~mask.unsqueeze(-1)                     # [B,T,1]
            if mask_behavior == "neg_inf":
                logits = logits.masked_fill(m, float("-inf"))
            elif mask_behavior == "zero":
                logits = logits.masked_fill(m, 0.0)
            elif mask_behavior != "none":
                raise ValueError("mask_behavior debe ser 'none' | 'neg_inf' | 'zero'.")
        return logits


# ----------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    # Pruebas rápidas
    torch.manual_seed(0)

    texts = ["hello", "helicopter"]
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_pad=True, add_bos=True, add_eos=True, add_unk=False)
    seqs = [vocab.encode(t, mode="char", add_bos=True, add_eos=True) for t in texts]
    tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)

    d_model = 32
    emb = TokenEmbedder(vocab_size=vocab.size, d_model=d_model, pos_kind="sinusoidal", tie_softmax=True, pad_id=vocab.pad_id)
    h0 = emb(tokens)                                # [B,T,d]
    # Tres variantes de máscara
    logits_none = emb.logits_from_hidden(h0, mask=None, mask_behavior="none")
    logits_zero = emb.logits_from_hidden(h0, mask=mask, mask_behavior="zero")
    logits_ni   = emb.logits_from_hidden(h0, mask=mask, mask_behavior="neg_inf")

    # CE con ignore_index (pad)
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:]
    targets[:, -1] = vocab.pad_id
    loss = F.cross_entropy(logits_none.view(-1, logits_none.size(-1)), targets.view(-1), ignore_index=vocab.pad_id)
    print(f"[tok_embed] CE tied = {loss.item():.3f}")

    # Factorizada
    emb_f = TokenEmbedder(vocab_size=vocab.size, d_model=48, pos_kind="sinusoidal",
                          tie_softmax=True, factor_rank=16, pad_id=vocab.pad_id)
    h0f = emb_f(tokens)
    logf = emb_f.logits_from_hidden(h0f, mask=mask, mask_behavior="zero")
    loss_f = F.cross_entropy(logf.view(-1, logf.size(-1)), targets.view(-1), ignore_index=vocab.pad_id)
    print(f"[tok_embed] CE tied (fact) = {loss_f.item():.3f}")
    print("[tok_embed] OK ✓")
