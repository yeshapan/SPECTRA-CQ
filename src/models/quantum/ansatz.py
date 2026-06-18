import pennylane as qml

def build_ansatz(weights, wires, topology="basic"):
    """
    Parameterized Quantum Circuit (VQC Ansatz).
    Dynamically implements different entanglement topologies for ablation studies.
    
    Args:
        weights (Tensor): Trainable angles for the rotation gates. 
                          Shape depends on topology:
                          - 'none' or 'basic': (layers, num_qubits)
                          - 'strong': (layers, num_qubits, 3)
        wires (Iterable): The target qubits.
        topology (str): Defines the entanglement mapping. 
                        Options: 'none', 'basic' or 'strong'
    """
    if topology == "none":
        # Brute-force/Naive approach: No entanglement.
        # Applies independent RX rotations. Proves if quantum correlation is even mathematically necessary.
        layers = weights.shape[0]
        for layer in range(layers):
            for i, wire in enumerate(wires):
                qml.RX(weights[layer, i], wires=wire)
                
    elif topology == "basic":
        # Localized entanglement baseline.
        # PennyLane's BasicEntanglerLayers natively applies RX rotations and ring-CNOTs
        qml.BasicEntanglerLayers(weights=weights, wires=wires)
        
    elif topology == "strong":
        # Optimized approach for highly coupled macroscopic frequencies.
        # Applies generalized Rot gates (RX, RY, RZ) and all-to-all CNOT combinations.
        qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        
    else:
        raise ValueError(f"Topology '{topology}' is not supported.")