import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from src.models.quantum.vqc import quantum_circuit, N_QUBITS

class VQCTorchLayer(nn.Module):
    """
    Wraps the PennyLane QNode into a native PyTorch Module.
    This enables the classical Adam optimizer to calculate gradients and 
    backpropagate them directly through the quantum rotations.
    """
    def __init__(self, n_layers=3, topology="basic"):
        super(VQCTorchLayer, self).__init__()
        self.n_layers = n_layers
        self.topology = topology
        
        # Define the exact parameter shape required by our ansatz
        # StronglyEntanglingLayers requires 3 Euler angles (RX, RY, RZ) per qubit per layer.
        # BasicEntanglerLayers and our naive 'none' approach only require 1 angle (RX).
        if self.topology == "strong":
            weight_shapes = {"weights": (n_layers, N_QUBITS, 3)}
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
            weight_shapes = {"weights": (n_layers, N_QUBITS)}
            # Non-Zero Initialization: 'none' and 'basic' topologies will suffer absolute zero gradients if initialized near 0 due to sin(0) = 0 derivatives.
            # We use a Uniform distribution to ensure gradients can flow back immediately.
            init_method = {
                "weights": lambda x: torch.nn.init.uniform_(x, a=-np.pi, b=np.pi)
            }
            
        # Fix: PennyLane 0.45+ strictly checks the signature.
        # Create a wrapper to hide the 'topology' argument from the TorchLayer shape checker.
        def circuit_wrapper(inputs, weights):
            # .func extracts the raw Python function from your original QNode
            # This prevents a "QNode-inside-a-QNode" crash while passing the static string
            return quantum_circuit.func(inputs, weights, topology=self.topology)
            
        # Dynamically re-compile the QNode using your original device
        dynamic_qnode = qml.QNode(circuit_wrapper, quantum_circuit.device)
        
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