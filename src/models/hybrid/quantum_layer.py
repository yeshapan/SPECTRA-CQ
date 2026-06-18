import torch
import torch.nn as nn
import pennylane as qml
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
        else:
            weight_shapes = {"weights": (n_layers, N_QUBITS)}
        
        # Initialize PennyLane's Torch bridge
        # We pass topology as a static keyword argument to the QNode
        self.vqc = qml.qnn.TorchLayer(
            quantum_circuit, 
            weight_shapes,
            topology=self.topology
        )

    def forward(self, x):
        """
        Executes the forward pass through the quantum circuit.
        """
        return self.vqc(x)