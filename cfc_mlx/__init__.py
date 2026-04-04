"""CfC-MLX: Closed-form Continuous-time neural networks for Apple MLX.

A native MLX implementation of CfC (Hasani et al., 2022) with
Neural Circuit Policy (NCP) sparse wiring (Lechner et al., 2020).

Usage:
    from cfc_mlx import CfC, CfCCell, AutoNCP

    wiring = AutoNCP(units=64, output_size=10, sparsity_level=0.5)
    model = CfC(input_size=32, hidden_size=64, wiring=wiring)

    output, hidden = model(x)  # x: [batch, seq_len, input_size]
"""

from .wiring import NCP, AutoNCP
from .cfc import CfCCell, CfC

__version__ = "0.1.0"
__all__ = ["CfCCell", "CfC", "NCP", "AutoNCP"]
