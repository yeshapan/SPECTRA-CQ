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
            weight_shapes
        )

    def forward(self, x):
        """
        Executes the forward pass through the quantum circuit.
        """
        return self.vqc(x)