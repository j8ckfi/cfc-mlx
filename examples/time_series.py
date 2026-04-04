"""Example: CfC for time series forecasting.

Generates a synthetic multi-channel signal, trains a CfC model
to predict future values, and compares against a persistence baseline.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cfc_mlx import CfC, AutoNCP


def generate_data(n_samples=5000, n_channels=3, seq_len=50, horizon=10):
    """Generate synthetic multi-channel oscillatory data."""
    t = np.linspace(0, 100, n_samples + seq_len + horizon)
    data = np.zeros((len(t), n_channels), dtype=np.float32)
    for ch in range(n_channels):
        freq = 0.1 + ch * 0.05
        phase = ch * np.pi / 3
        data[:, ch] = np.sin(2 * np.pi * freq * t + phase) + 0.1 * np.random.randn(len(t))

    # Create windows
    X, y = [], []
    for i in range(n_samples):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon])
    return np.array(X), np.array(y)


def main():
    print("Generating synthetic data...")
    X, y = generate_data()
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Model
    n_channels = 3
    hidden_size = 32
    horizon = 10
    wiring = AutoNCP(units=hidden_size, output_size=n_channels, sparsity_level=0.5)
    print(wiring.summary())

    cfc = CfC(
        input_size=n_channels,
        hidden_size=hidden_size,
        wiring=wiring,
        backbone_units=64,
        backbone_layers=1,
        return_sequences=False,
    )
    decoder = nn.Linear(cfc.state_size, horizon * n_channels)
    mx.eval(cfc.parameters(), decoder.parameters())

    optimizer = optim.Adam(learning_rate=1e-3)

    # Wrap CfC + decoder into a single module for clean gradient computation
    class Model(nn.Module):
        def __init__(self, cfc, decoder):
            super().__init__()
            self.cfc = cfc
            self.decoder = decoder

        def __call__(self, x):
            out, _ = self.cfc(x)
            return self.decoder(out).reshape(-1, horizon, n_channels)

    model = Model(cfc, decoder)
    mx.eval(model.parameters())

    x_ref, y_ref = [None], [None]
    batch_size = 64

    def loss_fn(model):
        pred = model(x_ref[0])
        return mx.mean((pred - y_ref[0]) ** 2)

    print("\nTraining...")
    for epoch in range(30):
        indices = np.random.permutation(len(X_train))
        losses = []
        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            x_ref[0] = mx.array(X_train[idx])
            y_ref[0] = mx.array(y_train[idx])

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(loss.item())

        # Validation
        x_v = mx.array(X_val)
        y_v = mx.array(y_val)
        pred_v = model(x_v)
        val_loss = mx.mean((pred_v - y_v) ** 2)
        mx.eval(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | train={np.mean(losses):.4f} val={val_loss.item():.4f}")

    # Persistence baseline
    persist = np.broadcast_to(X_val[:, -1:, :], y_val.shape)
    persist_mse = np.mean((persist - y_val) ** 2)
    print(f"\nCfC val MSE:         {val_loss.item():.4f}")
    print(f"Persistence MSE:     {persist_mse:.4f}")
    print(f"CfC / Persistence:   {val_loss.item() / persist_mse:.4f}")


if __name__ == "__main__":
    main()
