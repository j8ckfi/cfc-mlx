"""Closed-form Continuous-time (CfC) neural network in MLX.

Implements the CfC cell from Hasani et al. (2022) "Closed-form
Continuous-time Neural Networks" with optional NCP sparse wiring
from Lechner et al. (2020).

The key update rule:
    x = backbone(concat(input, hidden))
    ff1 = tanh(W1 @ x + b1)
    ff2 = tanh(W2 @ x + b2)
    t_interp = sigmoid(t_a * ts + t_b)
    hidden_new = ff1 * (1 - t_interp) + t_interp * ff2

This is a closed-form approximation to the Liquid Time-Constant (LTC)
ODE, eliminating the need for numerical ODE solvers while preserving
the continuous-time dynamics.

References:
    - Hasani et al., "Closed-form Continuous-time Neural Networks" (2022)
    - Hasani et al., "Liquid Time-constant Networks" (2021)
    - Lechner et al., "Neural Circuit Policies" (2020)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .wiring import NCP


def lecun_tanh(x):
    """LeCun's scaled tanh: 1.7159 * tanh(2/3 * x).

    Self-normalizing activation that maintains unit variance through
    layers when inputs have unit variance.
    """
    return 1.7159 * mx.tanh(0.6667 * x)


class CfCCell(nn.Module):
    """Single-step CfC cell.

    Processes one timestep: (input_t, hidden_{t-1}, ts) -> hidden_t

    The cell concatenates the input and previous hidden state, passes
    through a backbone network, then computes a time-dependent
    interpolation between two nonlinear projections.

    Args:
        input_size: dimension of input features per timestep
        hidden_size: dimension of hidden state (ignored if wiring provided)
        wiring: optional NCP wiring for sparse connectivity
        backbone_units: hidden dimension of backbone MLP layers
        backbone_layers: depth of backbone MLP
        backbone_dropout: dropout probability in backbone (0 = no dropout)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        wiring: Optional[NCP] = None,
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wiring = wiring

        if wiring is not None:
            wiring.build(input_size)
            self._state_size = wiring.state_size
        else:
            self._state_size = hidden_size

        # Backbone MLP: concat(input, hidden) -> features
        cat_size = input_size + self._state_size
        in_features = cat_size
        bb_linears = []
        for _ in range(backbone_layers):
            bb_linears.append(nn.Linear(in_features, backbone_units))
            in_features = backbone_units
        self.backbone_layers = bb_linears
        self.backbone_dropout = (
            nn.Dropout(backbone_dropout) if backbone_dropout > 0 else None
        )
        bb_out = backbone_units if backbone_layers > 0 else cat_size

        # CfC heads: two fixed-point attractors + time gate
        self.ff1 = nn.Linear(bb_out, self._state_size)
        self.ff2 = nn.Linear(bb_out, self._state_size)
        self.time_a = nn.Linear(bb_out, self._state_size)
        self.time_b = nn.Linear(bb_out, self._state_size)

    @property
    def state_size(self):
        """Dimension of the hidden state."""
        return self._state_size

    def _apply_backbone(self, x):
        for layer in self.backbone_layers:
            x = lecun_tanh(layer(x))
            if self.backbone_dropout is not None:
                x = self.backbone_dropout(x)
        return x

    def __call__(self, input_t, hx, ts=1.0):
        """Forward pass for a single timestep.

        Args:
            input_t: [batch, input_size] — current input
            hx: [batch, state_size] — previous hidden state
            ts: scalar, [batch], or [batch, 1] — time delta since last step.
                For uniformly sampled data, use ts=1.0 (default).
                For irregularly sampled data, pass actual time deltas.

        Returns:
            h_new: [batch, state_size] — updated hidden state
        """
        if isinstance(ts, (int, float)):
            ts = mx.array(ts, dtype=input_t.dtype)
        if ts.ndim == 0:
            ts = mx.expand_dims(ts, axis=0)
        if ts.ndim == 1:
            ts = mx.expand_dims(ts, axis=-1)

        x = mx.concatenate([input_t, hx], axis=-1)
        x = self._apply_backbone(x)

        ff1 = mx.tanh(self.ff1(x))
        ff2 = mx.tanh(self.ff2(x))
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = mx.sigmoid(t_a * ts + t_b)

        return ff1 * (1.0 - t_interp) + t_interp * ff2


class CfC(nn.Module):
    """CfC sequence model.

    Wraps CfCCell to process variable-length sequences, with optional
    output projection and mixed-memory gating.

    Args:
        input_size: per-timestep input dimension
        hidden_size: hidden state dimension (total NCP units if wiring given)
        output_size: if set, adds a linear projection on the output
        wiring: optional NCP wiring for sparse connectivity
        backbone_units: backbone MLP hidden dimension
        backbone_layers: backbone MLP depth
        backbone_dropout: backbone dropout rate
        return_sequences: True = all timesteps, False = last only
        mixed_memory: if True, use a sigmoid gate to blend old/new hidden state

    Example:
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> model = CfC(input_size=5, hidden_size=64, wiring=wiring)
        >>> x = mx.random.normal((batch, seq_len, 5))
        >>> output, final_hidden = model(x)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        wiring: Optional[NCP] = None,
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        return_sequences: bool = True,
        mixed_memory: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.mixed_memory = mixed_memory

        self.cell = CfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            wiring=wiring,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        self._state_size = self.cell.state_size

        if mixed_memory:
            self.mix_gate = nn.Linear(self._state_size, self._state_size)

        self.readout = nn.Linear(self._state_size, output_size) if output_size else None

    @property
    def state_size(self):
        """Dimension of the hidden state."""
        return self._state_size

    def __call__(self, x, timespans=None, initial_state=None):
        """Process a sequence through the CfC.

        Args:
            x: [batch, seq_len, input_size] — input sequence
            timespans: [batch, seq_len, 1] or None — time deltas between steps.
                If None, assumes uniform spacing (ts=1.0 for each step).
            initial_state: [batch, state_size] or None — initial hidden state.
                If None, starts from zeros.

        Returns:
            output: [batch, seq_len, dim] if return_sequences, else [batch, dim]
                where dim = output_size if readout exists, else state_size
            final_hidden: [batch, state_size] — final hidden state
        """
        batch_size, seq_len, _ = x.shape

        hx = initial_state if initial_state is not None else mx.zeros(
            (batch_size, self._state_size)
        )

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            ts = timespans[:, t, :] if timespans is not None else 1.0

            h_new = self.cell(input_t, hx, ts)

            if self.mixed_memory:
                gate = mx.sigmoid(self.mix_gate(h_new))
                hx = gate * hx + (1.0 - gate) * h_new
            else:
                hx = h_new

            outputs.append(hx)

        if self.return_sequences:
            output = mx.stack(outputs, axis=1)
        else:
            output = outputs[-1]

        if self.readout is not None:
            output = self.readout(output)

        return output, hx
