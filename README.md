# CfC-MLX

A native [Apple MLX](https://github.com/ml-explore/mlx) implementation of **Closed-form Continuous-time (CfC)** neural networks with **Neural Circuit Policy (NCP)** sparse wiring.

CfC networks are continuous-time recurrent models that handle irregular time series, long-range dependencies, and variable-length sequences. Unlike LTC (Liquid Time-Constant) networks which require ODE solvers, CfC uses a closed-form solution — making it fast and straightforward to implement.

This package brings CfC to Apple Silicon via MLX.

## Installation

```bash
pip install cfc-mlx
```

Or from source:

```bash
git clone https://github.com/j8ck/cfc-mlx.git
cd cfc-mlx
pip install -e .
```

Requires Python 3.10+ and MLX 0.20+.

## Quick start

```python
import mlx.core as mx
from cfc_mlx import CfC, AutoNCP

# Create sparse NCP wiring (64 neurons, 10 outputs)
wiring = AutoNCP(units=64, output_size=10, sparsity_level=0.5)

# Build CfC model
model = CfC(
    input_size=32,
    hidden_size=64,
    wiring=wiring,
    backbone_units=128,
    return_sequences=True,
)

# Forward pass
x = mx.random.normal((batch:=8, seq_len:=100, 32))
output, final_hidden = model(x)
# output: [8, 100, 64], final_hidden: [8, 64]
```

### Irregular time series

CfC naturally handles irregular sampling — pass time deltas between observations:

```python
timespans = mx.random.uniform(shape=(8, 100, 1)) * 2.0  # variable dt
output, hidden = model(x, timespans=timespans)
```

### With output projection

```python
model = CfC(
    input_size=32,
    hidden_size=64,
    output_size=10,       # adds linear readout
    return_sequences=False,  # last timestep only
)
output, hidden = model(x)  # output: [8, 10]
```

## Architecture

### CfC cell update

At each timestep, the cell computes:

```
x = backbone(concat(input_t, hidden_{t-1}))
ff1 = tanh(W1 @ x + b1)          # attractor 1
ff2 = tanh(W2 @ x + b2)          # attractor 2
t_interp = sigmoid(a * dt + b)    # time-dependent gate
hidden_t = ff1 * (1 - t_interp) + t_interp * ff2
```

The backbone uses LeCun tanh activation (1.7159 * tanh(2/3 * x)) for self-normalizing properties.

### NCP wiring

NCP defines a sparse, layered connectivity inspired by C. elegans:

```
sensory → inter → command ⟲ → motor
```

- **Sensory neurons** receive external input
- **Interneurons** provide intermediate processing
- **Command neurons** integrate with recurrent connections
- **Motor neurons** produce output

`AutoNCP` automatically configures layer sizes (40% command, 60% inter) and connection density from a sparsity parameter.

## Components

| Class | Description |
|-------|-------------|
| `CfCCell` | Single-step CfC update |
| `CfC` | Full sequence model wrapping CfCCell |
| `NCP` | Manual NCP wiring specification |
| `AutoNCP` | Automatic NCP configuration |

## References

- Hasani et al., ["Closed-form Continuous-time Neural Networks"](https://arxiv.org/abs/2106.13898) (2022)
- Hasani et al., ["Liquid Time-constant Networks"](https://arxiv.org/abs/2006.04439) (2021)
- Lechner et al., ["Neural Circuit Policies Enabling Auditable Autonomy"](https://doi.org/10.1038/s42256-020-00237-3) (2020)

## License

MIT
