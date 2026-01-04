"""
Module 3: Variational Quantum Algorithm (VQA) with CUDA-Q
==========================================================

We now implement a complete VQA training loop using CUDA-Q. This reimplements
the VQE example we did with PennyLane, but now using CUDA-Q's API.

We'll optimize a parameterized quantum circuit to find the ground state of
a Hamiltonian, demonstrating the full variational quantum algorithm workflow.

Author: Matías Bilkis
"""

try:
    import cudaq
except ImportError:
    raise ImportError(
        "CUDA-Q is not installed. Please install it following the instructions at: "
        "https://nvidia.github.io/cuda-quantum/latest/install.html"
    )

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_hamiltonian():
    """
    Create the Hamiltonian H = σ_x (for 1-qubit) or H = X₁X₂ + Z₁Z₂ (for 2-qubits).
    
    For simplicity, we'll work with the 1-qubit case first.
    
    Returns
    -------
    callable
        Function that computes expectation value of the Hamiltonian
    """
    @cudaq.kernel
    def measure_x(theta: float, phi: float):
        """
        Measure σ_x by applying Hadamard before measurement.
        """
        q = cudaq.qubit()
        ry(theta, q)
        rz(phi, q)
        h(q)
        mz(q)
    
    def expectation_value(theta, phi, shots=10000):
        """
        Compute expectation value of σ_x.
        
        Parameters
        ----------
        theta : float
            Rotation angle around y-axis
        phi : float
            Rotation angle around z-axis
        shots : int
            Number of measurement shots
            
        Returns
        -------
        float
            Expectation value ⟨σ_x⟩
        """
        result = cudaq.sample(measure_x, theta, phi, shots_count=shots)
        prob_0 = result.get('0', 0) / shots
        prob_1 = result.get('1', 0) / shots
        return prob_0 * 1 + prob_1 * (-1)
    
    return expectation_value


def parameter_shift_gradient(theta, phi, expectation_func, shift=np.pi/2, shots=10000):
    """
    Compute gradient using parameter-shift rule.
    
    ∂⟨H⟩/∂θ = (1/2) [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]
    
    Parameters
    ----------
    theta : float
        Current rotation angle around y-axis
    phi : float
        Current rotation angle around z-axis
    expectation_func : callable
        Function that computes expectation values
    shift : float
        Shift amount (typically π/2)
    shots : int
        Number of measurement shots
        
    Returns
    -------
    tuple
        (gradient w.r.t. theta, gradient w.r.t. phi)
    """
    # Gradient w.r.t. theta
    expval_plus = expectation_func(theta + shift, phi, shots=shots)
    expval_minus = expectation_func(theta - shift, phi, shots=shots)
    grad_theta = (expval_plus - expval_minus) / 2
    
    # Gradient w.r.t. phi
    expval_plus = expectation_func(theta, phi + shift, shots=shots)
    expval_minus = expectation_func(theta, phi - shift, shots=shots)
    grad_phi = (expval_plus - expval_minus) / 2
    
    return grad_theta, grad_phi


def gradient_descent_step(params, grads, lr=0.1):
    """
    Perform one step of gradient descent.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameters [theta, phi]
    grads : np.ndarray
        Gradients [dE/dtheta, dE/dphi]
    lr : float
        Learning rate
        
    Returns
    -------
    np.ndarray
        Updated parameters
    """
    return params - lr * grads


def train_vqa(n_iterations=100, learning_rate=0.01, shots=10000):
    """
    Train a Variational Quantum Algorithm to find the ground state.
    
    We optimize the parameters (θ, φ) to minimize the energy ⟨H⟩ = ⟨σ_x⟩.
    
    Parameters
    ----------
    n_iterations : int
        Number of optimization steps
    learning_rate : float
        Learning rate for gradient descent
    shots : int
        Number of measurement shots per gradient evaluation
        
    Returns
    -------
    tuple
        (trajectory_params, trajectory_energy, final_params)
    """
    print("=" * 70)
    print("Training Variational Quantum Algorithm with CUDA-Q")
    print("=" * 70)
    
    # Create expectation value function
    expectation_func = create_hamiltonian()
    
    # Initialize parameters
    params = np.array([np.pi/4, np.pi/2 + 0.1])
    trajectory_params = [params.copy()]
    trajectory_energy = []
    
    # Initial energy
    energy = expectation_func(params[0], params[1], shots=shots)
    trajectory_energy.append(energy)
    
    print(f"\nInitial parameters: θ = {params[0]:.4f}, φ = {params[1]:.4f}")
    print(f"Initial energy: {energy:.6f}")
    print(f"Target energy (ground state): -1.000000\n")
    
    for iteration in tqdm(range(n_iterations), desc="Optimizing"):
        # Compute gradients using parameter-shift rule
        grads = parameter_shift_gradient(
            params[0], params[1], expectation_func, shots=shots
        )
        
        # Update parameters
        params = gradient_descent_step(params, np.array(grads), lr=learning_rate)
        
        # Measure energy (use more shots for accurate tracking)
        energy = expectation_func(params[0], params[1], shots=shots)
        
        trajectory_params.append(params.copy())
        trajectory_energy.append(energy)
    
    print(f"\nFinal parameters: θ = {params[0]:.4f}, φ = {params[1]:.4f}")
    print(f"Final energy: {trajectory_energy[-1]:.6f}")
    print(f"Target energy (ground state): -1.000000")
    
    return np.array(trajectory_params), np.array(trajectory_energy), params


def plot_training_results(trajectory_params, trajectory_energy):
    """
    Visualize the optimization trajectory.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Energy vs iterations
    ax1.plot(trajectory_energy, linewidth=3, color="blue", alpha=0.7,
             label="CUDA-Q VQA training")
    ax1.axhline(y=-1, color='black', linestyle='--', linewidth=2,
                label="Ground state energy")
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Energy ⟨H⟩", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Energy Convergence", fontsize=16)
    
    # Parameter trajectory
    ax2.plot(trajectory_params[:, 0], label=r'$\theta$', linewidth=3, alpha=0.7)
    ax2.plot(trajectory_params[:, 1], label=r'$\phi$', linewidth=3, alpha=0.7)
    ax2.axhline(y=np.pi/2, color='red', linestyle='--', alpha=0.5,
                label=r'$\theta^* = \pi/2$')
    ax2.axhline(y=np.pi, color='green', linestyle='--', alpha=0.5,
                label=r'$\phi^* = \pi$')
    ax2.set_xlabel("Iteration", fontsize=14)
    ax2.set_ylabel("Parameter Value", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Parameter Evolution", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('cudaq_vqa_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'cudaq_vqa_training.png'")
    plt.close()


def two_qubit_vqe():
    """
    Implement VQE for a 2-qubit system: H = X₁X₂ + Z₁Z₂.
    
    This is more complex as we need to measure multiple Pauli terms.
    """
    print("\n" + "=" * 70)
    print("2-Qubit VQE with CUDA-Q: H = X₁X₂ + Z₁Z₂")
    print("=" * 70)
    
    @cudaq.kernel
    def ansatz(theta0: float, phi0: float, theta1: float, phi1: float,
               theta2: float, theta3: float):
        """
        Entangling ansatz for 2-qubit system.
        """
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        
        # Initial rotations
        ry(theta0, q0)
        rz(phi0, q0)
        ry(theta1, q1)
        rz(phi1, q1)
        
        # Entangling gate
        x.ctrl(q0, q1)
        
        # Final rotations
        ry(theta2, q0)
        ry(theta3, q1)
    
    @cudaq.kernel
    def measure_xx(theta0: float, phi0: float, theta1: float, phi1: float,
                   theta2: float, theta3: float):
        """
        Measure X₁X₂ by applying Hadamard to both qubits.
        """
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        ry(theta0, q0)
        rz(phi0, q0)
        ry(theta1, q1)
        rz(phi1, q1)
        x.ctrl(q0, q1)
        ry(theta2, q0)
        ry(theta3, q1)
        h(q0)
        h(q1)
        mz(q0)
        mz(q1)
    
    @cudaq.kernel
    def measure_zz(theta0: float, phi0: float, theta1: float, phi1: float,
                   theta2: float, theta3: float):
        """
        Measure Z₁Z₂ (direct measurement).
        """
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        ry(theta0, q0)
        rz(phi0, q0)
        ry(theta1, q1)
        rz(phi1, q1)
        x.ctrl(q0, q1)
        ry(theta2, q0)
        ry(theta3, q1)
        mz(q0)
        mz(q1)
    
    def expectation_value(params, shots=10000):
        """
        Compute ⟨H⟩ = ⟨X₁X₂⟩ + ⟨Z₁Z₂⟩.
        """
        # Measure X₁X₂
        result_xx = cudaq.sample(measure_xx, *params, shots_count=shots)
        expval_xx = compute_pauli_expectation(result_xx, 'XX')
        
        # Measure Z₁Z₂
        result_zz = cudaq.sample(measure_zz, *params, shots_count=shots)
        expval_zz = compute_pauli_expectation(result_zz, 'ZZ')
        
        return expval_xx + expval_zz
    
    def compute_pauli_expectation(result, pauli_type):
        """
        Compute expectation value from measurement results.
        
        For XX: eigenvalues are +1 for |00⟩,|11⟩ and -1 for |01⟩,|10⟩
        For ZZ: eigenvalues are +1 for |00⟩,|11⟩ and -1 for |01⟩,|10⟩
        """
        total = sum(result.values())
        expval = 0.0
        
        for state, count in result.items():
            prob = count / total
            if pauli_type == 'XX' or pauli_type == 'ZZ':
                # Both have same eigenvalues
                if state in ['00', '11']:
                    expval += prob * 1
                else:  # '01' or '10'
                    expval += prob * (-1)
        
        return expval
    
    # Initialize parameters
    params = np.zeros(6)
    
    print(f"\nInitial energy: {expectation_value(params):.6f}")
    print("Target energy (ground state): -2.000000")
    print("\nNote: Full optimization would require implementing gradient")
    print("      computation for multi-qubit case. This is a simplified example.")
    
    return expectation_value


if __name__ == "__main__":
    # Train 1-qubit VQA
    trajectory_params, trajectory_energy, final_params = train_vqa(
        n_iterations=50,  # Reduced for faster execution
        learning_rate=0.01,
        shots=5000  # Reduced for faster execution
    )
    
    # Plot results
    plot_training_results(trajectory_params, trajectory_energy)
    
    # 2-qubit example (simplified)
    two_qubit_vqe()
    
    print("\n" + "=" * 70)
    print("VQA training with CUDA-Q complete!")
    print("=" * 70)

