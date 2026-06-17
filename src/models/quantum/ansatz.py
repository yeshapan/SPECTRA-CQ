import pennylane as qml

def build_ansatz(weights, wires):
    """
    Parameterized Quantum Circuit (VQC Ansatz).
    Implements a hardware-efficient ring topology. Each layer consists of:
    1. Independent RX rotations parameterized by the current weights.
    2. A cascade of CNOT gates entangling qubit(i) with qubit(i+1).
    
    Args:
        weights (Tensor): Trainable angles for the rotation gates. Shape: (layers, num_qubits)
        wires (Iterable): The target qubits.
    """
    # PennyLane's BasicEntanglerLayers natively applies RX rotations and ring-CNOTs
    qml.BasicEntanglerLayers(weights=weights, wires=wires)