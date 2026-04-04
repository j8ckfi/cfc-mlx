"""Neural Circuit Policy (NCP) wiring for CfC networks.

Implements the sparse wiring topology from Lechner et al. (2020),
inspired by C. elegans connectome structure:
  sensory -> inter -> command -> motor

Adjacency matrix encodes connection existence and polarity (+/- 1).
"""

import numpy as np
import mlx.core as mx


class NCP:
    """Neural Circuit Policy wiring specification.

    Defines which neurons connect to which, with excitatory/inhibitory
    polarity. The wiring is used by CfCCell to mask weight matrices,
    enforcing sparse, biologically-inspired connectivity.

    The architecture has four neuron layers:
    - Sensory neurons: receive external input
    - Interneurons: intermediate processing
    - Command neurons: integrate information, with recurrent connections
    - Motor neurons: produce output

    Args:
        inter_neurons: number of interneurons
        command_neurons: number of command neurons
        motor_neurons: number of motor neurons (= output dimension)
        sensory_fanout: max connections per sensory neuron to inter layer
        inter_fanout: max connections per inter neuron to command layer
        recurrent_command: max recurrent connections per command neuron
        motor_fanin: max connections per motor neuron from command layer
        seed: random seed for reproducible wiring
    """

    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int,
        inter_fanout: int,
        recurrent_command: int,
        motor_fanin: int,
        seed: int = 42,
    ):
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command = recurrent_command
        self.motor_fanin = motor_fanin
        self.seed = seed

        self.sensory_neurons = 0
        self.num_neurons = 0
        self.adjacency = None
        self.sensory_adjacency = None
        self._built = False

    @property
    def state_size(self):
        """Hidden state dimension = inter + command + motor."""
        return self.inter_neurons + self.command_neurons + self.motor_neurons

    @property
    def output_size(self):
        """Output dimension = motor neurons."""
        return self.motor_neurons

    def build(self, input_dim: int):
        """Build adjacency matrices given input dimension.

        Creates two matrices:
        - sensory_adjacency: [input_dim, state_size] — input-to-hidden
        - adjacency: [state_size, state_size] — recurrent hidden-to-hidden

        Connections are sparse with random +/-1 polarity (excitatory/inhibitory).

        Args:
            input_dim: number of input features (= number of sensory neurons)
        """
        if self._built and self.sensory_neurons == input_dim:
            return
        self.sensory_neurons = input_dim
        self.num_neurons = input_dim + self.state_size
        rng = np.random.RandomState(self.seed)

        n_inter = self.inter_neurons
        n_cmd = self.command_neurons
        n_motor = self.motor_neurons

        # Offsets within hidden state
        inter_start = 0
        cmd_start = n_inter
        motor_start = n_inter + n_cmd

        # --- Sensory -> Inter ---
        self.sensory_adjacency = np.zeros(
            (input_dim, self.state_size), dtype=np.float32
        )
        for src in range(input_dim):
            targets = rng.choice(
                n_inter, size=min(self.sensory_fanout, n_inter), replace=False
            )
            for t in targets:
                self.sensory_adjacency[src, inter_start + t] = rng.choice([-1, 1])

        # --- Recurrent connections ---
        self.adjacency = np.zeros(
            (self.state_size, self.state_size), dtype=np.float32
        )

        # Inter -> Command
        for src in range(n_inter):
            targets = rng.choice(
                n_cmd, size=min(self.inter_fanout, n_cmd), replace=False
            )
            for t in targets:
                self.adjacency[inter_start + src, cmd_start + t] = rng.choice([-1, 1])

        # Command recurrent
        for src in range(n_cmd):
            targets = rng.choice(
                n_cmd, size=min(self.recurrent_command, n_cmd), replace=False
            )
            for t in targets:
                self.adjacency[cmd_start + src, cmd_start + t] = rng.choice([-1, 1])

        # Command -> Motor
        for dst in range(n_motor):
            sources = rng.choice(
                n_cmd, size=min(self.motor_fanin, n_cmd), replace=False
            )
            for s in sources:
                self.adjacency[cmd_start + s, motor_start + dst] = rng.choice([-1, 1])

        self._built = True

    def get_masks(self):
        """Return binary masks (1 where connected, 0 otherwise).

        Returns:
            sensory_mask: mx.array [input_dim, state_size]
            recurrent_mask: mx.array [state_size, state_size]
        """
        assert self._built, "Call build(input_dim) first"
        return (
            mx.array(np.abs(self.sensory_adjacency)),
            mx.array(np.abs(self.adjacency)),
        )

    def get_polarities(self):
        """Return signed adjacency matrices (+1 excitatory, -1 inhibitory).

        Returns:
            sensory_polarity: mx.array [input_dim, state_size]
            recurrent_polarity: mx.array [state_size, state_size]
        """
        assert self._built, "Call build(input_dim) first"
        return mx.array(self.sensory_adjacency), mx.array(self.adjacency)

    def summary(self) -> str:
        """Return a human-readable summary of the wiring."""
        lines = [
            f"NCP Wiring:",
            f"  Sensory:  {self.sensory_neurons}",
            f"  Inter:    {self.inter_neurons}",
            f"  Command:  {self.command_neurons}",
            f"  Motor:    {self.motor_neurons}",
            f"  State:    {self.state_size}",
        ]
        if self._built:
            s_dens = np.mean(np.abs(self.sensory_adjacency) > 0)
            r_dens = np.mean(np.abs(self.adjacency) > 0)
            lines.append(f"  Sensory density:   {s_dens:.1%}")
            lines.append(f"  Recurrent density: {r_dens:.1%}")
        return "\n".join(lines)


class AutoNCP(NCP):
    """Automatic NCP wiring configuration.

    Given total neuron count and output size, automatically determines
    layer sizes and fanout following the 40/60 command/inter split
    from the reference implementation.

    Args:
        units: total number of hidden neurons (inter + command + motor)
        output_size: number of motor (output) neurons
        sparsity_level: fraction of possible connections to omit (0-1).
            0.0 = fully connected, 1.0 = no connections.
        seed: random seed for reproducible wiring
    """

    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: int = 42,
    ):
        n_motor = output_size
        remaining = units - n_motor
        if remaining <= 0:
            raise ValueError(
                f"units ({units}) must be greater than output_size ({output_size})"
            )
        n_command = max(int(0.4 * remaining), 1)
        n_inter = remaining - n_command

        density = 1.0 - sparsity_level
        sensory_fanout = max(int(density * n_inter), 1)
        inter_fanout = max(int(density * n_command), 1)
        recurrent_command = max(int(density * n_command), 1)
        motor_fanin = max(int(density * n_command), 1)

        super().__init__(
            inter_neurons=n_inter,
            command_neurons=n_command,
            motor_neurons=n_motor,
            sensory_fanout=sensory_fanout,
            inter_fanout=inter_fanout,
            recurrent_command=recurrent_command,
            motor_fanin=motor_fanin,
            seed=seed,
        )
        self.sparsity_level = sparsity_level
