from __future__ import annotations

from hpm.model import choose_local_attention_memory_policy

GIB = 1024**3


def _policy(total_gib: int, free_gib: int):
    return choose_local_attention_memory_policy(
        requested_mode="auto",
        seq_len=512,
        batch_size=32,
        heads=4,
        radius=256,
        head_dim=48,
        element_size=4,
        device_type="cuda",
        free_bytes=free_gib * GIB,
        total_bytes=total_gib * GIB,
    )


def test_auto_8gib_gpu_uses_memory_saver_immediately():
    p = _policy(8, 7)
    assert p["mode"] == "memory_saver"
    assert p["chunk_size"] == 32
    assert p["checkpoint"] is True


def test_auto_a100_40gib_first_layer_can_use_speed_path():
    # Modal telemetry from the failed run showed ~38.85 GiB free entering the
    # first local-attention layer. Auto should keep that layer fast.
    p = _policy(40, 39)
    assert p["mode"] == "speed"
    assert p["chunk_size"] == 512
    assert p["checkpoint"] is False


def test_auto_a100_40gib_later_layer_switches_to_memory_saver():
    # The same run showed ~26.74 GiB free entering the next local-attention
    # layer after the first layer's autograd state had been retained. That is
    # no longer enough once backward reserve is included.
    p = _policy(40, 27)
    assert p["mode"] == "memory_saver"
    assert p["chunk_size"] == 32
    assert p["checkpoint"] is True


def test_auto_80gib_device_can_keep_speed_with_same_geometry():
    p = _policy(80, 68)
    assert p["mode"] == "speed"
    assert p["checkpoint"] is False


def test_manual_speed_override_wins():
    p = choose_local_attention_memory_policy(
        requested_mode="speed",
        seq_len=512,
        batch_size=32,
        heads=4,
        radius=256,
        head_dim=48,
        element_size=4,
        device_type="cuda",
        free_bytes=1 * GIB,
        total_bytes=8 * GIB,
    )
    assert p["mode"] == "speed"
    assert p["checkpoint"] is False


def test_manual_memory_saver_override_wins():
    p = choose_local_attention_memory_policy(
        requested_mode="memory_saver",
        seq_len=512,
        batch_size=32,
        heads=4,
        radius=256,
        head_dim=48,
        element_size=4,
        device_type="cuda",
        free_bytes=39 * GIB,
        total_bytes=40 * GIB,
    )
    assert p["mode"] == "memory_saver"
    assert p["checkpoint"] is True


def test_cpu_auto_avoids_checkpoint_recompute():
    p = choose_local_attention_memory_policy(
        requested_mode="auto",
        seq_len=512,
        batch_size=32,
        heads=4,
        radius=256,
        head_dim=48,
        element_size=4,
        device_type="cpu",
    )
    assert p["mode"] == "speed"
    assert p["checkpoint"] is False


def test_speed_and_memory_saver_paths_are_numerically_equivalent(monkeypatch):
    import torch
    from hpm.model import LocalCausalSelfAttention

    torch.manual_seed(7)
    module = LocalCausalSelfAttention(d_model=16, heads=4, window=8, dropout=0.0)
    x1 = torch.randn(2, 24, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    monkeypatch.setenv("HPM_ATTENTION_MEMORY_MODE", "speed")
    y1 = module(x1)
    loss1 = y1.square().mean()
    loss1.backward()
    grad1 = x1.grad.detach().clone()

    module.zero_grad(set_to_none=True)
    module._attention_memory_policy_reported = True
    monkeypatch.setenv("HPM_ATTENTION_MEMORY_MODE", "memory_saver")
    y2 = module(x2)
    loss2 = y2.square().mean()
    loss2.backward()
    grad2 = x2.grad.detach().clone()

    torch.testing.assert_close(y1, y2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(grad1, grad2, rtol=0.0, atol=0.0)
