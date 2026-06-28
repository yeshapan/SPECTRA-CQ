import pennylane as qml
from .embedding import embed_features
from .ansatz import build_ansatz

# Multi-sensor fusion: 6 sensors * 2 features per sensor = 12 qubits
N_QUBITS = 12

# Initialize the state-vector simulator
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights, topology="basic"):
    """
    The Master QNode. Stitches the data embedding, parameterized ansatz, and measurement together.
    
    Args:
        inputs (Tensor): The 12-dimensional continuous classical vector.
        weights (Tensor): Trainable angles for the rotation gates.
        topology (str): Defines the entanglement mapping ('none', 'basic', 'strong', 'bridge_graph_hard', 'bridge_graph_soft').
    
    Returns:
        List of expectation values (Pauli-Z) to collapse the quantum state back to continuous classical floats.
    """
    embed_features(inputs, wires=range(N_QUBITS))
    build_ansatz(weights, wires=range(N_QUBITS), topology=topology)
    
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]