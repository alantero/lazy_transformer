# data/dataloaders.py
# Simple datasets and collate functions for windowed LM training.
# - SequenceDataset: wraps token id lists
# - collate_batch: pads/clamps to a common length
# - build_collar_mask: marks overlap (“collar”) tokens given (W, O)
# - make_dataloader: convenience factory
#
# Pure PyTorch; no global dependencies on other project files.

from __future__ import annotations
from typing import List, Dict, Iterable, Optional, Tuple
import math
import torch
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor


# ------------------------------- dataset --------------------------------------

class SequenceDataset(Dataset):
    """
    Tiny wrapper around a list of token-id sequences.

    Args:
      sequences: list of lists with token IDs (Python ints)
    """
    def __init__(self, sequences: List[List[int]]):
        super().__init__()
        self._seqs = [list(map(int, s)) for s in sequences]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> List[int]:
        return self._seqs[idx]


# ------------------------------- collate --------------------------------------

def _add_bos_eos(
    seq: List[int],
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> List[int]:
    if bos_id is not None:
        seq = [int(bos_id)] + seq
    if eos_id is not None:
        seq = seq + [int(eos_id)]
    return seq


def collate_batch(
    batch: List[List[int]],
    *,
    pad_id: int = 0,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    max_len: Optional[int] = None,
    clamp_left: bool = False,
) -> Dict[str, Tensor]:
    """
    Collate variable-length token sequences to a padded batch.

    Args:
      batch: list of token-id lists
      pad_id: padding token id
      bos_id/eos_id: optional special tokens to prepend/append per sample
      max_len: if set, clamp sequences to this length
      clamp_left: if True and clamping is needed, keep the *right* part (recent tokens)

    Returns:
      dict with:
        tokens  [B, T] Long
        mask    [B, T] Bool (True for valid tokens)
        targets [B, T] Long (copy of tokens by default; shift is done in the loss)
    """
    proc = []
    for s in batch:
        s2 = _add_bos_eos(list(s), bos_id, eos_id)
        if max_len is not None and len(s2) > max_len:
            if clamp_left:
                s2 = s2[-max_len:]
            else:
                s2 = s2[:max_len]
        proc.append(s2)

    B = len(proc)
    T = max(len(s) for s in proc) if proc else 0
    tokens = torch.full((B, T), int(pad_id), dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)

    for i, s in enumerate(proc):
        n = len(s)
        tokens[i, :n] = torch.tensor(s, dtype=torch.long)
        mask[i, :n] = True

    batch_out = {
        "tokens": tokens,
        "mask": mask,
        "targets": tokens.clone(),  # next-token shift happens downstream
    }
    return batch_out


# ------------------------------ collar mask -----------------------------------

def build_collar_mask(
    lengths: Iterable[int],
    *,
    W: int,
    O: int,
) -> Tensor:
    """
    Compute a boolean mask [B, T_max] that marks tokens lying in *overlaps*
    when slicing each sequence into windows of width W and overlap O.
    Overlaps are the last O positions of each window (except the final) and the
    first O positions of the subsequent window; we mark each token position once.

    We assume windows advance by stride S = W - O. If the last window would
    exceed the sequence length, we conceptually pad to the right (like the
    training loop with pad=True) and intersect with the true [0, T) range.

    Args:
      lengths: iterable of per-sample lengths (ints)
      W: window length
      O: overlap length, 0 <= O < W

    Returns:
      Bool tensor of shape [B, T_max] where T_max = max(lengths)
    """
    lens = list(map(int, lengths))
    if len(lens) == 0:
        return torch.zeros((0, 0), dtype=torch.bool)

    B = len(lens)
    T_max = max(lens)
    m = torch.zeros((B, T_max), dtype=torch.bool)
    if O <= 0:
        return m

    S = W - O
    if S <= 0:
        # Degenerate; treat all tokens as collar to be safe.
        return torch.ones((B, T_max), dtype=torch.bool)

    for b, T in enumerate(lens):
        # Compute window start indices with right-padding behavior
        # so that the last start is the largest s with s < T.
        starts = []
        s = 0
        while s < max(T, 1):  # ensure at least one window for T>0
            starts.append(s)
            s += S

        # Mark the overlap region between consecutive windows
        for i in range(len(starts) - 1):
            left_end = starts[i] + W       # exclusive end of window i
            ol_start = max(starts[i + 1], left_end - O)
            # In practice, with stride S=W-O, ol_start == left_end - O.
            ol_end = min(left_end, T)      # clip to true length
            if ol_start < ol_end:
                m[b, ol_start:ol_end] = True
    return m


# ----------------------------- convenience API --------------------------------

def make_dataloader(
    sequences: List[List[int]],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_id: int = 0,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    max_len: Optional[int] = None,
    clamp_left: bool = False,
) -> DataLoader:
    """
    Build a DataLoader over raw token sequences with our collate.

    Returns batches shaped like collate_batch() plus 'collar_mask' (for the batch),
    which you can compute using build_collar_mask later given (W, O).
    We do not precompute collar here because (W, O) are model-config dependent.
    """
    ds = SequenceDataset(sequences)

    def _collate(samples: List[List[int]]) -> Dict[str, Tensor]:
        batch = collate_batch(
            samples, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
            max_len=max_len, clamp_left=clamp_left,
        )
        return batch

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate
    )


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    print("[dataloaders] Running sanity tests...")
    # Toy samples with heterogeneous lengths
    seqs = [
        [3, 4, 5, 6, 7, 8, 9, 10],
        [3, 4, 5, 6],
        [3, 4, 5, 6, 7, 8, 9],
    ]
    pad_id = 0
    bos_id, eos_id = 1, 2

    batch = collate_batch(seqs, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, max_len=None)
    tokens, mask = batch["tokens"], batch["mask"]
    B, T = tokens.shape
    print(f"  collate: tokens shape={tuple(tokens.shape)}, mask sum={int(mask.sum())}")

    # Collar mask sanity for a setting that tiles exactly
    W, O = 6, 2
    lens = mask.sum(dim=1).tolist()  # actual per-sample lengths after BOS/EOS
    collar = build_collar_mask(lens, W=W, O=O)
    assert collar.shape == tokens.shape
    # There should be some collar tokens if sequences are long enough
    any_collar = bool(collar.any().item())
    print(f"  collar: shape={tuple(collar.shape)}, any={any_collar}")
    # Check collars subset of valid tokens
    assert ((~mask) & collar).sum().item() == 0

    # DataLoader smoke
    loader = make_dataloader(seqs, batch_size=2, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    one = next(iter(loader))
    print(f"  loader: batch tokens shape={tuple(one['tokens'].shape)}")

    print("[dataloaders] All good ✓")
