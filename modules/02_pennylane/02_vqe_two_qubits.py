"""
Module 2 - Part 2: Variational Quantum Eigensolver (VQE) with Two Qubits
=========================================================================

We now move to a 2-qubit system to explore entanglement and more complex
quantum circuits. We consider the Hamiltonian H = X₁X₂ + Z₁Z₂, whose
ground state is the singlet state |ψ⟩ = (|01⟩ - |10⟩)/√2.

We compare separable circuits (no entanglement) with entangling circuits
(using CNOT gates) to demonstrate the importance of entanglement in
quantum algorithms.

Author: Matías Bilkis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import pennylane as qml
except ImportError:
    raise ImportError(
        "PennyLane is not installed. Please install it with: pip install pennylane"
    )


def verify_singlet_state():
    """
    Verify that the singlet state is indeed the ground state of H = XX + ZZ.
    
    The singlet state is |ψ⟩ = (|01⟩ - |10⟩)/√2 with energy E₀ = -2.
    """
    print("=" * 70)
    print("Verifying the Singlet State")
    print("=" * 70)
    
    # Get Pauli matrices from PennyLane
    X = qml.PauliX.matrix
    Z = qml.PauliZ.matrix
    
    # Construct Hamiltonian: H = X₁X₂ + Z₁Z₂
    Hamiltonian = np.kron(X, X) + np.kron(Z, Z)
    
    # Diagonalize to find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
    
    print(f"\nEigenvalues (energies): {eigenvalues}")
    print(f"Ground state energy: E₀ = {eigenvalues[0]:.6f}")
    
    # Ground state (first column of eigenvectors)
    ground_state = eigenvectors[:, 0]
    print(f"\nGround state (first 4 components): {ground_state[:4]}")
    
    # Verify it's the singlet
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    overlap = np.abs(np.dot(np.conj(ground_state), singlet))**2
    print(f"\nOverlap with singlet state: |⟨singlet|ground⟩|² = {overlap:.6f}")
    print("(Should be close to 1.0)")
    
    # Verify energy
    def expected_value(observable, state):
        return np.dot(np.conj(state), np.dot(observable, state))
    
    energy_singlet = expected_value(Hamiltonian, singlet)
    print(f"\nEnergy of singlet state: ⟨singlet|H|singlet⟩ = {energy_singlet:.6f}")
    
    return Hamiltonian, singlet


def separable_circuit(params, **kwargs):
    """
    Separable circuit: no entanglement between qubits.
    
    This circuit applies independent rotations on each qubit:
    - RY(θ₀), RZ(φ₀) on qubit 0
    - RY(θ₁), RZ(φ₁) on qubit 1
    
    Parameters
    ----------
    params : array-like
        [θ₀, φ₀, θ₁, φ₁] - rotation angles
    """
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RZ(params[3], wires=1)


def entangled_circuit(params, **kwargs):
    """
    Entangling circuit: uses CNOT to create quantum correlations.
    
    This circuit:
    1. Applies local rotations on both qubits
    2. Applies CNOT gate (entangles the qubits)
    3. Applies additional local rotations
    
    Parameters
    ----------
    params : array-like
        [θ₀, φ₀, θ₁, φ₁, θ₂, θ₃] - rotation angles
    """
    # Initial local rotations
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RZ(params[3], wires=1)
    
    # CNOT gate (creates entanglement)
    qml.CNOT(wires=[0, 1])
    
    # Additional local rotations
    qml.RY(params[4], wires=0)
    qml.RY(params[5], wires=1)


def train_vqe_circuits(n_steps=100, learning_rate=0.1):
    """
    Train both separable and entangled circuits to find the ground state.
    
    Parameters
    ----------
    n_steps : int
        Number of optimization steps
    learning_rate : float
        Learning rate for gradient descent
        
    Returns
    -------
    tuple
        (costs_evolution, final_params_separable, final_params_entangled)
    """
    print("\n" + "=" * 70)
    print("Training VQE Circuits")
    print("=" * 70)
    
    # Define Hamiltonian: H = X₁X₂ + Z₁Z₂
    coeffs = [1.0, 1.0]
    obs = [
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliZ(0) @ qml.PauliZ(1)
    ]
    H_vqe = qml.Hamiltonian(coeffs, obs)
    
    # Create devices
    quantum_device_separable = qml.device('default.qubit', wires=2, shots=1000)
    quantum_device_entangled = qml.device('default.qubit', wires=2, shots=1000)
    
    # Create cost functions using PennyLane's VQE module
    # Note: In newer versions of PennyLane, we use qml.ExpvalCost or qml.ExpvalCost
    # For the latest API, we'll use the qnode approach
    @qml.qnode(quantum_device_separable)
    def cost_separable(params):
        separable_circuit(params)
        return qml.expval(H_vqe)
    
    @qml.qnode(quantum_device_entangled)
    def cost_entangled(params):
        entangled_circuit(params)
        return qml.expval(H_vqe)
    
    # Initialize parameters (all zeros = identity)
    params_separable = np.zeros(4)
    params_entangled = np.zeros(6)
    
    # Optimizers
    opt_separable = qml.GradientDescentOptimizer(stepsize=learning_rate)
    opt_entangled = qml.GradientDescentOptimizer(stepsize=learning_rate)
    
    costs_evolution = []
    
    print(f"\nInitial energy (separable): {cost_separable(params_separable):.6f}")
    print(f"Initial energy (entangled): {cost_entangled(params_entangled):.6f}")
    print(f"Target energy (ground state): -2.000000\n")
    
    for i in tqdm(range(n_steps), desc="Optimizing"):
        params_separable = opt_separable.step(cost_separable, params_separable)
        params_entangled = opt_entangled.step(cost_entangled, params_entangled)
        
        costs_evolution.append([
            cost_separable(params_separable),
            cost_entangled(params_entangled)
        ])
    
    costs_evolution = np.array(costs_evolution)
    
    print(f"\nFinal energy (separable): {costs_evolution[-1, 0]:.6f}")
    print(f"Final energy (entangled): {costs_evolution[-1, 1]:.6f}")
    print(f"Target energy (ground state): -2.000000")
    
    return costs_evolution, params_separable, params_entangled


def plot_training_comparison(costs_evolution):
    """
    Plot the training curves for both circuits.
    """
    plt.figure(figsize=(16, 8))
    
    plt.plot(costs_evolution[:, 0], linewidth=4, color="red", alpha=0.7,
             label="Separable Circuit")
    plt.plot(costs_evolution[:, 1], linewidth=4, color="blue", alpha=0.7,
             label="Entangling Circuit")
    plt.axhline(y=-2, color='black', linestyle='--', linewidth=2,
                label="Ground state energy")
    
    plt.xlabel("Gradient Descent Steps", fontsize=16)
    plt.ylabel("Cost function (energy)", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.title("VQE Training: Separable vs Entangling Circuits", fontsize=18)
    
    plt.tight_layout()
    plt.savefig('vqe_training_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'vqe_training_comparison.png'")
    plt.close()


def visualize_optimized_circuit(params_entangled):
    """
    Visualize the optimized entangling circuit.
    """
    print("\n" + "=" * 70)
    print("Optimized Entangling Circuit")
    print("=" * 70)
    
    quantum_device = qml.device('default.qubit', wires=2, shots=1000)
    
    @qml.qnode(quantum_device)
    def entangled_circuit_visualization(params):
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        qml.RY(params[2], wires=1)
        qml.RZ(params[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[4], wires=0)
        qml.RY(params[5], wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]
    
    drawer = qml.draw(entangled_circuit_visualization)
    print("\nCircuit diagram:")
    print(drawer(params=params_entangled))
    
    print(f"\nOptimized parameters:")
    print(f"  θ₀ = {params_entangled[0]:.4f}")
    print(f"  φ₀ = {params_entangled[1]:.4f}")
    print(f"  θ₁ = {params_entangled[2]:.4f}")
    print(f"  φ₁ = {params_entangled[3]:.4f}")
    print(f"  θ₂ = {params_entangled[4]:.4f}")
    print(f"  θ₃ = {params_entangled[5]:.4f}")


if __name__ == "__main__":
    # Verify singlet state
    Hamiltonian, singlet = verify_singlet_state()
    
    # Train circuits
    costs_evolution, params_sep, params_ent = train_vqe_circuits(
        n_steps=100,
        learning_rate=0.1
    )
    
    # Plot results
    plot_training_comparison(costs_evolution)
    
    # Visualize optimized circuit
    visualize_optimized_circuit(params_ent)
    
    print("\n" + "=" * 70)
    print("Key Insight: Entanglement is crucial for preparing the ground state!")
    print("The separable circuit cannot create the singlet state, which requires")
    print("quantum correlations between the two qubits.")
    print("=" * 70)

