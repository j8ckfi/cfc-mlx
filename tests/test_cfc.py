"""Tests for CfC-MLX implementation."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from cfc_mlx import CfCCell, CfC, NCP, AutoNCP
from cfc_mlx.cfc import lecun_tanh


class TestLeCunTanh:
    def test_zero(self):
        assert abs(lecun_tanh(mx.array([0.0])).item()) < 1e-6

    def test_derivative_at_origin(self):
        eps = 1e-4
        dy = (lecun_tanh(mx.array([eps])).item() - lecun_tanh(mx.array([0.0])).item()) / eps
        assert abs(dy - 1.1439) < 0.01


class TestNCP:
    def test_state_size(self):
        w = NCP(8, 6, 3, 4, 3, 2, 3)
        assert w.state_size == 17
        assert w.output_size == 3

    def test_build_shapes(self):
        w = NCP(8, 6, 3, 4, 3, 2, 3)
        w.build(5)
        s_mask, r_mask = w.get_masks()
        assert s_mask.shape == (5, 17)
        assert r_mask.shape == (17, 17)

    def test_sparsity(self):
        w = NCP(8, 6, 3, 4, 3, 2, 3)
        w.build(5)
        s_mask, r_mask = w.get_masks()
        assert 0 < mx.mean(s_mask > 0).item() < 1.0
        assert 0 < mx.mean(r_mask > 0).item() < 1.0

    def test_sensory_only_to_inter(self):
        w = NCP(8, 6, 3, 4, 3, 2, 3)
        w.build(5)
        s_mask, _ = w.get_masks()
        assert mx.sum(s_mask[:, 8:]).item() == 0  # no direct sensory->cmd/motor

    def test_reproducible(self):
        w1 = NCP(8, 6, 3, 4, 3, 2, 3, seed=123)
        w1.build(5)
        w2 = NCP(8, 6, 3, 4, 3, 2, 3, seed=123)
        w2.build(5)
        assert np.array_equal(w1.adjacency, w2.adjacency)


class TestAutoNCP:
    def test_layer_sizes(self):
        a = AutoNCP(64, 5)
        assert a.motor_neurons == 5
        assert a.state_size == 64

    def test_small_units(self):
        with pytest.raises(ValueError):
            AutoNCP(5, 5)


class TestCfCCell:
    def test_output_shape(self):
        cell = CfCCell(5, 16, backbone_units=32)
        mx.eval(cell.parameters())
        h = cell(mx.random.normal((4, 5)), mx.zeros((4, 16)))
        mx.eval(h)
        assert h.shape == (4, 16)

    def test_with_ncp(self):
        w = AutoNCP(32, 5)
        cell = CfCCell(5, 32, wiring=w)
        mx.eval(cell.parameters())
        h = cell(mx.random.normal((4, 5)), mx.zeros((4, 32)))
        mx.eval(h)
        assert h.shape == (4, 32)

    def test_no_nans(self):
        cell = CfCCell(5, 16)
        mx.eval(cell.parameters())
        h = cell(mx.random.normal((4, 5)), mx.zeros((4, 16)))
        mx.eval(h)
        assert not mx.any(mx.isnan(h)).item()


class TestCfC:
    def test_sequence_output(self):
        model = CfC(5, 16, output_size=3, return_sequences=True)
        mx.eval(model.parameters())
        out, h = model(mx.random.normal((4, 20, 5)))
        mx.eval(out, h)
        assert out.shape == (4, 20, 3)
        assert h.shape == (4, 16)

    def test_last_only(self):
        model = CfC(5, 16, output_size=3, return_sequences=False)
        mx.eval(model.parameters())
        out, _ = model(mx.random.normal((4, 20, 5)))
        mx.eval(out)
        assert out.shape == (4, 3)

    def test_no_readout(self):
        model = CfC(5, 16, return_sequences=False)
        mx.eval(model.parameters())
        out, _ = model(mx.random.normal((4, 20, 5)))
        mx.eval(out)
        assert out.shape == (4, 16)

    def test_mixed_memory(self):
        model = CfC(5, 16, mixed_memory=True, return_sequences=True)
        mx.eval(model.parameters())
        out, _ = model(mx.random.normal((4, 10, 5)))
        mx.eval(out)
        assert out.shape == (4, 10, 16)

    def test_timespans(self):
        model = CfC(5, 16, return_sequences=True)
        mx.eval(model.parameters())
        x = mx.random.normal((4, 20, 5))
        ts = mx.random.uniform(shape=(4, 20, 1)) * 2.0
        out, _ = model(x, timespans=ts)
        mx.eval(out)
        assert out.shape == (4, 20, 16)

    def test_initial_state(self):
        model = CfC(5, 16, return_sequences=False)
        mx.eval(model.parameters())
        h0 = mx.random.normal((4, 16))
        out, h = model(mx.random.normal((4, 10, 5)), initial_state=h0)
        mx.eval(out, h)
        assert out.shape == (4, 16)

    def test_with_ncp_wiring(self):
        w = AutoNCP(32, 5)
        model = CfC(5, 32, wiring=w, return_sequences=True)
        mx.eval(model.parameters())
        out, _ = model(mx.random.normal((4, 20, 5)))
        mx.eval(out)
        assert out.shape == (4, 20, 32)

    def test_gradient_flow(self):
        model = CfC(5, 16, output_size=3, return_sequences=False)
        mx.eval(model.parameters())

        x = mx.random.normal((4, 10, 5))
        y = mx.random.normal((4, 3))

        def loss_fn(model):
            out, _ = model(x)
            return mx.mean((out - y) ** 2)

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)

        assert not mx.isnan(loss).item()
        # Check at least some gradients are nonzero
        has_grad = False
        def _check(tree):
            nonlocal has_grad
            if isinstance(tree, dict):
                for v in tree.values():
                    _check(v)
            elif isinstance(tree, list):
                for v in tree:
                    _check(v)
            elif isinstance(tree, mx.array):
                if mx.any(tree != 0).item():
                    has_grad = True
        _check(grads)
        assert has_grad


class TestPyTorchComparison:
    def test_shapes_match(self):
        """Compare output shapes with PyTorch ncps (if installed)."""
        try:
            import torch
            from ncps.torch import CfC as TorchCfC
            from ncps.wirings import AutoNCP as TorchAutoNCP
        except ImportError:
            pytest.skip("PyTorch/ncps not installed")

        x_np = np.random.randn(4, 20, 5).astype(np.float32)

        torch_wiring = TorchAutoNCP(32, 5, sparsity_level=0.5)
        torch_model = TorchCfC(5, torch_wiring, batch_first=True)
        torch_model.eval()
        with torch.no_grad():
            out_torch = torch_model(torch.from_numpy(x_np))[0].numpy()

        mlx_wiring = AutoNCP(32, 5, sparsity_level=0.5)
        mlx_model = CfC(5, 32, wiring=mlx_wiring, return_sequences=True)
        mx.eval(mlx_model.parameters())
        out_mlx, _ = mlx_model(mx.array(x_np))
        mx.eval(out_mlx)
        out_mlx = np.array(out_mlx)

        assert out_torch.shape[0] == out_mlx.shape[0]
        assert out_torch.shape[1] == out_mlx.shape[1]
        assert np.all(np.isfinite(out_mlx))
