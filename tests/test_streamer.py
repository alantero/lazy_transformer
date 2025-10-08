# tests/test_streamer.py
# Streamer early-stop on local optimum/plateau using a dummy constant-entropy model.

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, vocab=32, peak=5.0):
        super().__init__()
        self.vocab = vocab
        # fixed logits peaked at token 0
        base = torch.zeros(vocab)
        base[0] = float(peak)
        self.register_buffer("_logits", base.view(1, 1, -1))  # [1,1,V]

    def forward_tokens(self, tokens, mask=None):
        B, T = tokens.shape
        return self._logits.expand(B, T, self.vocab).clone()

def test_streamer_stops_on_plateau():
    from infer.streamer import Streamer, StreamerConfig
    torch.manual_seed(0)

    B, T0 = 2, 5
    tokens = torch.zeros(B, T0, dtype=torch.long)
    mask = torch.ones(B, T0, dtype=torch.bool)
    model = DummyModel(vocab=32, peak=6.0).eval()

    cfg = StreamerConfig(
        W=16, O=4, temperature=1.0, top_k=0, top_p=1.0,
        stop_on_optimum=True, opt_window=5, opt_patience=3, opt_eps=1e-6, min_new_tokens=2,
        teleport_every=None, teleport_on_low_capacity=False,
    )
    gen = Streamer(model, cfg)
    out = gen.generate(tokens, mask, max_new_tokens=32)
    # Constant entropy ⇒ no improvement ⇒ should stop early
    assert out["stopped_optimum"] is True
    assert out["generated"].shape[1] >= 2  # produced at least min_new_tokens
