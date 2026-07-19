import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# We pull embedding and ansatz directly to dynamically compile the QNode 
# without relying on the hardcoded Phase 4 constraints in vqc.py
from src.models.quantum.embedding import embed_features
from src.models.quantum.ansatz import build_ansatz

class VQCTorchLayer(nn.Module):
    """
    Wraps the PennyLane QNode into a native PyTorch Module.
    This enables the classical Adam optimizer to calculate gradients and 
    backpropagate them directly through the quantum rotations.
    """
    def __init__(self, n_layers=3, topology="basic", n_qubits=12):
        super().__init__() # Modern + safe syntax to avoid namespace reload errors
        self.n_layers = n_layers
        self.topology = topology
        self.n_qubits = n_qubits

        # Define the exact parameter shape required by our ansatz
        # StronglyEntanglingLayers requires 3 Euler angles (RX, RY, RZ) per qubit per layer.
        # BasicEntanglerLayers and our naive 'none' approach only require 1 angle (RX).
        if self.topology == "strong":
            weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
            # Identity Block Initialization: Required for deep entanglement scrambling
            init_method = {
                "weights": lambda x: torch.nn.init.normal_(x, mean=0.0, std=0.01)
            }

        elif "bridge_graph" in self.topology:
            # Multi-sensor fusion update: Physics-Informed Inter-Sensor Combinations
            # 6 sensors yield 15 unique spatial pair combinations: (6 * 5) / 2 = 15
            # We need exactly 1 parameter (CRY angle) per spatial pair per layer
            weight_shapes = {"weights": (n_layers, 15)}
            # Near-Zero Initialization: Allows the spatial connections to start as weak perturbations rather than chaotic cross-talk.
            # This enables Adam to slowly build the graph edge weights.
            init_method = {
                "weights": lambda x: torch.nn.init.uniform_(x, a=-0.1, b=0.1)
            }

        else:
            weight_shapes = {"weights": (n_layers, self.n_qubits)}
            # Non-Zero Initialization: 'none' and 'basic' topologies will suffer absolute zero gradients if initialized near 0 due to sin(0) = 0 derivatives.
            # We use a Uniform distribution to ensure gradients can flow back immediately.
            init_method = {
                "weights": lambda x: torch.nn.init.uniform_(x, a=-np.pi, b=np.pi)
            }
            
        # Dynamically re-compile the QNode using your original device limits
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def dynamic_qnode(inputs, weights):
            embed_features(inputs, wires=range(self.n_qubits))
            build_ansatz(weights, wires=range(self.n_qubits), topology=self.topology)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
            
        # Initialize PennyLane's Torch bridge with the clean signature
        self.vqc = qml.qnn.TorchLayer(
            dynamic_qnode, 
            weight_shapes,
            init_method=init_method  # Pass the new initialization strategy
        )

    def forward(self, x):
        """
        Executes the forward pass through the quantum circuit.
        """
        # Fix: The Classical-to-Quantum Bridge Latent Scaling
        # Bounding the classical logits to [-pi, pi] prevents the sinusoidal Ry embedding gradients from washing out or oscillating to zero.
        x_scaled = torch.tanh(x) * np.pi
        
        return self.vqc(x_scaled)