"""Regression test for the episodic_only router-leak bug.

Found 2026-08-02 while cross-checking uncommitted local changes: in
HpmLiteV2Model.__init__, `self.router` was set inside the
`if config.episodic_only: ... else: ...` branch (None vs. a real
HpmV2PathRouter), then unconditionally overwritten two lines later by a
second `self.router = HpmV2PathRouter(...)` outside the branch -- silently
undoing the None case. episodic_only=True therefore still allocated (and,
because forward() creates optimizer param groups from all named
parameters, trained) a full unused router module. forward() itself was
never affected -- it branches on `self.config.episodic_only` directly,
not on `self.router is None` -- so historical ablation evidence
numbers are NOT invalidated by this, but the ablation's stated claim
("removes the three unused components") was false about param count.

Never caught because no test exercised episodic_only=True at all.
"""
import torch

from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model


def _build_batch(bsz=2, seq_len=24, num_facts=4, query_tok=1, answer_tok=2):
    input_ids = torch.randint(5, 50, (bsz, seq_len))
    starts = torch.linspace(0, seq_len - 6, num_facts).long()
    mem_pos = torch.stack([starts, starts + 1], dim=-1)[None].expand(bsz, num_facts, 2).contiguous()
    mem_mask = torch.ones(bsz, num_facts, dtype=torch.bool)
    qkp = torch.full((bsz,), seq_len - 2, dtype=torch.long)
    input_ids[:, -2] = query_tok
    input_ids[:, -1] = answer_tok
    return input_ids, mem_pos, mem_mask, qkp


def test_episodic_only_does_not_allocate_a_router():
    cfg = HpmLiteV2Config(d_model=16, layers=1, heads=2, window=8, block_size=4, episodic_only=True)
    model = HpmLiteV2Model(cfg)
    assert model.router is None, (
        "episodic_only=True must not allocate a router -- if this fails, the "
        "unconditional `self.router = HpmV2PathRouter(...)` overwrite is back"
    )
    assert model.selective_recurrent is None
    assert model.fast_memory is None


def test_episodic_only_forward_uses_only_episodic_path():
    torch.manual_seed(0)
    cfg = HpmLiteV2Config(d_model=16, layers=1, heads=2, window=8, block_size=4, episodic_only=True)
    model = HpmLiteV2Model(cfg)
    input_ids, mem_pos, mem_mask, qkp = _build_batch()

    out = model(
        input_ids, mem_pos, mem_mask, answer_positions=None, query_key_positions=qkp,
        top_k=1, task="kv",
    )
    assert out["logits"].shape == (input_ids.size(0), input_ids.size(1), cfg.vocab_size)
    assert "router_weights" not in out["retrieval"], (
        "episodic_only forward must not report router_weights -- there is no router"
    )


def test_non_episodic_only_still_gets_a_real_router():
    cfg = HpmLiteV2Config(d_model=16, layers=1, heads=2, window=8, block_size=4, episodic_only=False)
    model = HpmLiteV2Model(cfg)
    assert model.router is not None
    assert model.selective_recurrent is not None
    assert model.fast_memory is not None
