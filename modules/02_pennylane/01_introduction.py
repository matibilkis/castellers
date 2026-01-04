"""
Module 2 - Part 2: Introduction to PennyLane
==============================================

Now that we understand the fundamentals "under the hood", we move to
PennyLane, a powerful quantum machine learning library. PennyLane is
platform-agnostic, meaning it can work with different quantum backends
(IBM, Google, Xanadu, etc.) and integrates seamlessly with classical ML
frameworks like PyTorch and TensorFlow.

Author: Matías Bilkis
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import pennylane as qml
except ImportError:
    raise ImportError(
        "PennyLane is not installed. Please install it with: pip install pennylane"
    )


def demonstrate_pennylane_basics():
    """
    Demonstrate basic PennyLane operations, revisiting our 1-qubit example.
    
    We reconstruct the same objective function and gradients we computed
    with Castellers, but now using PennyLane's elegant API.
    """
    print("=" * 70)
    print("PennyLane Basics: Revisiting Our 1-Qubit Example")
    print("=" * 70)
    
    # Define quantum device
    # "default.qubit" is PennyLane's built-in simulator
    quantum_device = qml.device('default.qubit', wires=1, shots=None)
    
    @qml.qnode(quantum_device)
    def one_qubit_hamiltonian_x(theta, phi):
        """
        Quantum circuit for our 1-qubit problem.
        
        The @qml.qnode decorator connects this function to the quantum device.
        This is how PennyLane elegantly handles the quantum-classical interface.
        """
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        return qml.expval(qml.PauliX(0))
    
    print("\nCircuit defined! Let's test it with some parameters...\n")
    
    # Test with different parameters
    test_params = [
        (0, 0),
        (np.pi/2, np.pi),
        (np.pi/4, np.pi/2)
    ]
    
    for theta, phi in test_params:
        energy = one_qubit_hamiltonian_x(theta, phi)
        print(f"θ = {theta:6.3f}, φ = {phi:6.3f}  →  ⟨H⟩ = {energy:8.6f}")
    
    return one_qubit_hamiltonian_x


def compare_shots_pennyLane():
    """
    Compare how different numbers of shots affect the accuracy of estimates.
    
    This reproduces the analysis we did with Castellers, but using PennyLane.
    """
    print("\n" + "=" * 70)
    print("Effect of Number of Shots (PennyLane)")
    print("=" * 70)
    
    phis = np.linspace(0, 2*np.pi, 100)
    plt.figure(figsize=(20, 6))
    
    for k, shots in enumerate([10, 1000, None]):
        # Create device with specified number of shots
        quantum_device = qml.device('default.qubit', wires=1, shots=shots)
        
        @qml.qnode(quantum_device)
        def one_qubit_hamiltonian_x(theta, phi):
            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=0)
            return qml.expval(qml.PauliX(0))
        
        ax = plt.subplot(1, 3, k+1)
        fun_str = str(shots) if isinstance(shots, int) else "infinite"
        ax.set_title(f"Number of shots = {fun_str}", fontsize=14)
        
        for theta, color, label in zip(
            [0, np.pi/4, np.pi/2],
            ["green", "blue", "red"],
            [r'$\theta = 0$', r'$\theta = \frac{\pi}{4}$', r'$\theta = \frac{\pi}{2}$']
        ):
            energies = [one_qubit_hamiltonian_x(theta, phi) for phi in phis]
            ax.plot(phis, energies, color=color, linewidth=3, alpha=0.75, label=label)
        
        ax.set_xticks(np.arange(0, 2*np.pi + np.pi/2, np.pi/2))
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.legend(fontsize=12)
        ax.set_xlabel(r'$\phi$', fontsize=14)
        ax.set_ylabel(r'$\langle H \rangle(\theta, \phi)$', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pennylane_shots_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'pennylane_shots_comparison.png'")
    plt.close()


def demonstrate_automatic_differentiation():
    """
    Demonstrate PennyLane's automatic differentiation capabilities.
    
    PennyLane can automatically compute gradients using the parameter-shift
    rule (or other methods) without us having to manually implement it.
    """
    print("\n" + "=" * 70)
    print("Automatic Differentiation with PennyLane")
    print("=" * 70)
    
    quantum_device = qml.device('default.qubit', wires=1, shots=1000)
    
    @qml.qnode(quantum_device, diff_method="parameter-shift")
    def one_qubit_hamiltonian_x(theta, phi):
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        return qml.expval(qml.PauliX(0))
    
    # PennyLane can automatically compute gradients!
    # dcircuit(theta, phi) returns [d⟨H⟩/dθ, d⟨H⟩/dφ]
    dcircuit = qml.grad(one_qubit_hamiltonian_x)
    
    print("\nTesting automatic differentiation...")
    test_theta, test_phi = np.pi/4, np.pi/2
    grads = dcircuit(test_theta, test_phi)
    
    print(f"\nAt θ = {test_theta:.4f}, φ = {test_phi:.4f}:")
    print(f"  d⟨H⟩/dθ = {grads[0]:.6f}")
    print(f"  d⟨H⟩/dφ = {grads[1]:.6f}")
    
    # Compare with analytical gradients
    # For ⟨H⟩ = sin(θ) cos(φ):
    # d⟨H⟩/dθ = cos(θ) cos(φ)
    # d⟨H⟩/dφ = -sin(θ) sin(φ)
    grad_theta_analytical = np.cos(test_theta) * np.cos(test_phi)
    grad_phi_analytical = -np.sin(test_theta) * np.sin(test_phi)
    
    print(f"\nAnalytical gradients:")
    print(f"  d⟨H⟩/dθ = {grad_theta_analytical:.6f}")
    print(f"  d⟨H⟩/dφ = {grad_phi_analytical:.6f}")
    
    print(f"\nErrors:")
    print(f"  Δ(d⟨H⟩/dθ) = {abs(grads[0] - grad_theta_analytical):.6f}")
    print(f"  Δ(d⟨H⟩/dφ) = {abs(grads[1] - grad_phi_analytical):.6f}")
    
    return dcircuit


if __name__ == "__main__":
    # Demonstrate basic operations
    one_qubit_hamiltonian_x = demonstrate_pennylane_basics()
    
    # Compare shots
    compare_shots_pennyLane()
    
    # Demonstrate automatic differentiation
    dcircuit = demonstrate_automatic_differentiation()
    
    print("\n" + "=" * 70)
    print("PennyLane makes quantum machine learning much easier!")
    print("=" * 70)

