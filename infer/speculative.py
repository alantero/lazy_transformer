# infer/speculative.py
# Speculative decoding (Phase 10.2): draft→verify with 2–3 adaptive steps.
# - Draft (student) proposes k tokens; Teacher verifies sequentialmente.
# - k se adapta con la “confianza” (entropía) del draft: 2–3 pasos.
# - Rechazo: si teacher no acepta, se muestrea del teacher en ese punto y se corta la verificación.
#
# Requisitos:
#   - Modelos compatibles con `ContinuousLM.forward_tokens(tokens, mask) -> [B,T,V]`.
#   - Tokenización SimpleVocab (char/word) para el sanity del __main__.
#
# Notas:
#   - Implementación robusta y simple (batch soportado iterando por elemento).
#   - Sampling con temperatura + top-k/top-p (núcleo autocontenido).
#   - Métricas: ratio de aceptación y tokens propuestos/verificados por iteración.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ safe imports ----------------------------------
# Permite ejecutar como script: python infer/speculative.py
if __package__ in (None, ""):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.loop import ContinuousLM, LoopConfig
try:
    from data.tok_embed import SimpleVocab, pack_batch  # preferido
except Exception:
    from data.tokenize import SimpleVocab, pack_batch   # fallback


Tensor = torch.Tensor


# --------------------------------- helpers ------------------------------------

def _last_indices_from_mask(mask: Tensor) -> Tensor:
    """
    mask: [B,T] bool (True=token válido). Devuelve [B] con el índice del último token válido.
    Si no hay máscara, se asume que todos los T son válidos (manejado fuera).
    """
    # idx = (longitud válida) - 1
    lengths = mask.to(dtype=torch.int64).sum(dim=1)  # [B]
    idx = (lengths - 1).clamp_min(0)
    return idx


@torch.no_grad()
def _last_logits(model: ContinuousLM, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    Devuelve logits en la última posición válida de cada secuencia: [B,V].
    """
    model.eval()
    logits = model.forward_tokens(tokens, mask)  # [B,T,V]
    B, T, V = logits.shape
    if mask is None:
        last_idx = torch.full((B,), T - 1, dtype=torch.long, device=logits.device)
    else:
        last_idx = _last_indices_from_mask(mask)
    out = logits[torch.arange(B, device=logits.device), last_idx]  # [B,V]
    return out


def _top_k_top_p_filtering(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """
    Filtra logits con top-k y/o top-p (nucleus). Mantiene forma [B,V].
    """
    B, V = logits.shape
    filtered = logits.clone()

    # top-k
    if top_k > 0 and top_k < V:
        kth_vals = torch.topk(filtered, top_k, dim=-1).values[:, -1].unsqueeze(-1)  # [B,1]
        mask_k = filtered < kth_vals
        filtered[mask_k] = -float("inf")

    # top-p
    if top_p < 1.0:
        # Ordenar por probabilidad
        probs = F.softmax(filtered, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)  # [B,V]
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # máscara donde el acumulado excede p (excepto al menos 1)
        mask_p = cum_probs > top_p
        mask_p[:, 0] = False
        # devolver a índice original
        scatter_mask = torch.zeros_like(mask_p, dtype=torch.bool)
        scatter_mask.scatter_(dim=-1, index=sorted_idx, src=mask_p)
        filtered[scatter_mask] = -float("inf")

    return filtered


def _sample_from_logits(logits: Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    logits [B,V] → sample ids [B], y probs escogidas [B] bajo softmax( logits/temperature ).
    Aplica top-k/top-p antes de muestrear. Devuelve (idx, prob_escolhida).
    """
    z = logits / float(max(temperature, 1e-8))
    zf = _top_k_top_p_filtering(z, top_k=top_k, top_p=top_p)
    p = F.softmax(zf, dim=-1)
    idx = torch.multinomial(p, num_samples=1).squeeze(-1)  # [B]
    prob = p.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [B]
    return idx, prob


def _entropy_last(logits_last: Tensor) -> Tensor:
    """
    Entropía por secuencia en NATs de las distribuciones [B,V] (posición última).
    """
    logp = F.log_softmax(logits_last, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)  # [B]
    return H


# ---------------------------- speculative engine ------------------------------

@dataclass
class SpeculativeConfig:
    # draft steps (adaptativos)
    steps_low: int = 2
    steps_high: int = 3
    entropy_threshold: float = 0.45   # en proporción de log(V): si H/log(V) <= thr → usar steps_high

    # sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # miscelánea
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_verify_per_iter: Optional[int] = None  # si quieres capar verificación por iteración (None = sin tope)


class SpeculativeGenerator:
    """
    Implementa Speculative Sampling con student (draft) + teacher (verificador).
    """
    def __init__(self, student: ContinuousLM, teacher: ContinuousLM, cfg: SpeculativeConfig):
        self.student = student.eval()
        self.teacher = teacher.eval()
        self.cfg = cfg

    @torch.no_grad()
    def _choose_steps(self, logits_last: Tensor) -> int:
        """
        Elige 2 o 3 pasos según entropía normalizada H/log(V).
        """
        B, V = logits_last.shape
        H = _entropy_last(logits_last)                # [B]
        h_norm = H / math.log(float(V))
        # criterio batch: si mayoría “confiada”, usa steps_high
        confident = (h_norm <= self.cfg.entropy_threshold).float().mean().item()
        return self.cfg.steps_high if confident >= 0.5 else self.cfg.steps_low

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        mask: Optional[Tensor],
        *,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        tokens [B,T], mask [B,T] bool. Devuelve:
          { 'tokens': nuevos tokens concatenados [B,T+N],
            'accept_rate': float,
            'proposed': int total propuestos,
            'accepted': int aceptados,
            'iterations': int }
        """
        device = tokens.device
        B, T = tokens.shape
        assert mask is None or mask.shape == (B, T)
        temp = self.cfg.temperature if temperature is None else float(temperature)
        k_top = self.cfg.top_k if top_k is None else int(top_k)
        p_top = self.cfg.top_p if top_p is None else float(top_p)

        # contadores
        total_proposed = 0
        total_accepted = 0
        iters = 0

        # bucle de generación
        cur_tokens = tokens.clone()
        cur_mask = mask.clone() if mask is not None else torch.ones_like(tokens, dtype=torch.bool, device=device)
        new_tokens: List[Tensor] = []

        while len(new_tokens) < max_new_tokens:
            iters += 1

            # -------- Draft propone k tokens (autoregresivo) --------
            logits_last = _last_logits(self.student, cur_tokens, cur_mask)  # [B,V]
            k = self._choose_steps(logits_last)
            proposed_ids: List[Tensor] = []
            proposed_probs: List[Tensor] = []

            # Hacemos k pasos con el draft (cada paso usa contexto extendido)
            for _ in range(k):
                last_logits_d = _last_logits(self.student, cur_tokens, cur_mask)  # [B,V]
                nxt, prob = _sample_from_logits(last_logits_d, temperature=temp, top_k=k_top, top_p=p_top)
                proposed_ids.append(nxt)    # [B]
                proposed_probs.append(prob) # [B]
                # actualizar contexto draft
                cur_tokens, cur_mask = _append_tokens(cur_tokens, cur_mask, nxt)
                total_proposed += B
                # si ya llegamos a max_new, paramos la propuesta
                if len(new_tokens) + len(proposed_ids) >= max_new_tokens:
                    break

            # Deshacer la extensión temporal de contexto para verificar (teacher arranca desde antes del primer propuesto)
            # → volvemos al estado previo a proponer (quitamos los k tokens añadidos en draft).
            steps_added = len(proposed_ids)
            cur_tokens = cur_tokens[:, :T + len(new_tokens)]  # cortar back al contexto verificado
            cur_mask = cur_mask[:, :T + len(new_tokens)]

            if steps_added == 0:
                break  # nada que verificar

            # -------- Teacher verifica secuencialmente --------
            accepted_this_iter = 0
            for i in range(steps_added):
                # logits del teacher en el último contexto verificado
                last_logits_t = _last_logits(self.teacher, cur_tokens, cur_mask)  # [B,V]
                # prob del token propuesto por draft
                prop_id = proposed_ids[i]    # [B]
                pd = F.softmax(_top_k_top_p_filtering(last_logits_t * 0 + 1.0, top_k=0, top_p=1.0), dim=-1)  # dummy to fix jit (not used)
                # Usamos p_draft almacenado en el paso correspondiente (ojo: debe ser bajo la dist del DRAFT)
                p_draft = proposed_probs[i]  # [B]

                # prob del teacher para el token propuesto
                p_teacher = F.softmax(last_logits_t / temp, dim=-1).gather(dim=-1, index=prop_id.unsqueeze(-1)).squeeze(-1)  # [B]

                # aceptación: a = min(1, p_t / p_d)
                ratio = (p_teacher / p_draft.clamp_min(1e-12)).clamp(max=1.0)
                u = torch.rand_like(ratio)
                accept_mask = (u <= ratio)

                # para aceptados: añadimos el propuesto
                if accept_mask.all():
                    cur_tokens, cur_mask = _append_tokens(cur_tokens, cur_mask, prop_id)
                    new_tokens.append(prop_id)
                    total_accepted += B
                    accepted_this_iter += 1
                    # continuar con el siguiente propuesto
                else:
                    # manejar por elemento de batch
                    #  - donde se acepta: usa prop_id
                    #  - donde se rechaza: muestrea del teacher y detener verificación
                    add_ids = torch.empty_like(prop_id)
                    # muestreo teacher
                    filtered = _top_k_top_p_filtering(last_logits_t / temp, top_k=k_top, top_p=p_top)
                    teacher_idx = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1).squeeze(-1)
                    add_ids[accept_mask] = prop_id[accept_mask]
                    add_ids[~accept_mask] = teacher_idx[~accept_mask]

                    cur_tokens, cur_mask = _append_tokens(cur_tokens, cur_mask, add_ids)
                    new_tokens.append(add_ids)
                    total_accepted += int(accept_mask.sum().item())  # cuenta aceptados (rechazados no)
                    # corto verificación al primer rechazo (especulativa estándar)
                    break

                # tope opcional de verificación por iteración
                if (self.cfg.max_verify_per_iter is not None) and (accepted_this_iter >= self.cfg.max_verify_per_iter):
                    break

            # Si aceptó todos los propuestos sin rechazo, se añade 1 token adicional del teacher (propiedad estándar)
            if accepted_this_iter == steps_added and (len(new_tokens) < max_new_tokens):
                last_logits_t = _last_logits(self.teacher, cur_tokens, cur_mask)  # [B,V]
                extra_id, _ = _sample_from_logits(last_logits_t, temperature=temp, top_k=k_top, top_p=p_top)
                cur_tokens, cur_mask = _append_tokens(cur_tokens, cur_mask, extra_id)
                new_tokens.append(extra_id)

        # concatenar resultados
        if new_tokens:
            # Each element in new_tokens is [B]; stacking on dim=1 yields [B, N]
            add = torch.stack(new_tokens, dim=1)  # [B, N]
            out_tokens = torch.cat([tokens, add], dim=1)
            base_mask = mask if mask is not None else torch.ones_like(tokens, dtype=torch.bool)
            out_mask = torch.cat(
                [base_mask, torch.ones(add.size(0), add.size(1), dtype=torch.bool, device=tokens.device)],
                dim=1,
            )
        else:
            out_tokens = tokens
            out_mask = mask if mask is not None else torch.ones_like(tokens, dtype=torch.bool)

        accept_rate = (total_accepted / max(total_proposed, 1)) if total_proposed > 0 else 1.0
        return {
            "tokens": out_tokens,
            "mask": out_mask,
            "accept_rate": float(accept_rate),
            "proposed": int(total_proposed),
            "accepted": int(total_accepted),
            "iterations": iters,
        }


def _append_tokens(tokens: Tensor, mask: Tensor, next_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Añade `next_ids` [B] al final de cada secuencia en `tokens` [B,T].
    Actualiza la máscara.
    """
    B = tokens.shape[0]
    next_ids = next_ids.view(B, 1)
    new_tokens = torch.cat([tokens, next_ids], dim=1)
    next_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
    new_mask = torch.cat([mask, next_mask], dim=1)
    return new_tokens, new_mask


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[speculative] Running sanity test...")

    # Mini vocab & datos
    texts = ["hello there", "general kenobi", "hello hello"]
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
    enc = lambda s: vocab.encode(s, mode="char", add_bos=True, add_eos=False)
    dec = lambda ids: vocab.decode(ids, mode="char")

    # Prompt
    prompts = ["hello "]
    seqs = [enc(s) for s in prompts]
    tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)  # [B,T]

    # Config modelos (compartida)
    cfg = LoopConfig(
        W=16, O=4, vocab_size=vocab.size, d_model=64, groups=8,
        cheb_deg=6, cheb_laplacian="cycle",
        skew_rank=8, R_rank=4,
        steps=2, dt=0.5, method="heun",
        tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
        pad_id=vocab.pad_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_gauge=True,
    )
    device = torch.device(cfg.device)

    # Teacher y student (para demo, iguales; en producción, teacher puede ser mayor)
    teacher = ContinuousLM(cfg).to(device).eval()
    student = ContinuousLM(cfg).to(device).eval()
    # (Opcional) cargar pesos reales:
    # teacher.load_state_dict(torch.load("teacher.pt", map_location=device))
    # student.load_state_dict(torch.load("student.pt", map_location=device))

    tokens = tokens.to(device)
    mask = mask.to(device)

    # Generación especulativa
    spec_cfg = SpeculativeConfig(steps_low=2, steps_high=3, entropy_threshold=0.45, temperature=1.0, top_k=0, top_p=1.0, device=str(device))
    gen = SpeculativeGenerator(student, teacher, spec_cfg)
    out = gen.generate(tokens, mask, max_new_tokens=24)

    # Mostrar
    out_tokens: Tensor = out["tokens"].cpu()
    for i in range(out_tokens.shape[0]):
        text_out = dec(out_tokens[i].tolist())
        print(f"  sample[{i}] accept={out['accept_rate']:.2%} len={out_tokens.shape[1]} → {text_out!r}")

    print("[speculative] Done ✓")
