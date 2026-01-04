"""
Module 3: CUDA-Q Basics
========================

CUDA-Q is NVIDIA's quantum computing platform, designed for high-performance
quantum simulation and execution. It provides a C++ and Python API for building
quantum programs that can run on simulators or actual quantum hardware.

In this module, we introduce the basics of CUDA-Q and demonstrate how to
build simple quantum circuits.

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


def basic_quantum_circuit():
    """
    Demonstrate basic quantum circuit operations with CUDA-Q.
    
    We'll create a simple circuit that prepares a superposition state
    and measures it.
    """
    print("=" * 70)
    print("CUDA-Q Basics: Simple Quantum Circuit")
    print("=" * 70)
    
    @cudaq.kernel
    def simple_circuit():
        """
        A simple quantum circuit that creates a superposition.
        
        This circuit:
        1. Applies Hadamard to create superposition
        2. Applies a rotation
        3. Measures the qubit
        """
        q = cudaq.qubit()
        h(q)  # Hadamard gate
        ry(0.5, q)  # Rotation around y-axis
        mz(q)  # Measure in Z basis
    
    # Execute the circuit
    print("\nExecuting quantum circuit...")
    result = cudaq.sample(simple_circuit, shots_count=1000)
    
    print("\nMeasurement results:")
    print(result)
    
    # Get probabilities
    print("\nProbabilities:")
    for state, count in result.items():
        prob = count / 1000.0
        print(f"  |{state}⟩: {prob:.4f} ({count} counts)")
    
    return result


def parameterized_circuit():
    """
    Demonstrate parameterized quantum circuits with CUDA-Q.
    
    We'll create a circuit with trainable parameters, similar to what
    we did with Castellers and PennyLane.
    """
    print("\n" + "=" * 70)
    print("CUDA-Q: Parameterized Quantum Circuit")
    print("=" * 70)
    
    @cudaq.kernel
    def parameterized_circuit(theta: float, phi: float):
        """
        Parameterized quantum circuit: Ry(θ) Rz(φ)
        
        Parameters
        ----------
        theta : float
            Rotation angle around y-axis
        phi : float
            Rotation angle around z-axis
        """
        q = cudaq.qubit()
        ry(theta, q)
        rz(phi, q)
        mz(q)
    
    # Test with different parameters
    test_params = [
        (0.0, 0.0),
        (np.pi/2, np.pi),
        (np.pi/4, np.pi/2)
    ]
    
    print("\nTesting different parameter values:")
    for theta, phi in test_params:
        result = cudaq.sample(parameterized_circuit, theta, phi, shots_count=1000)
        prob_0 = result.get('0', 0) / 1000.0
        prob_1 = result.get('1', 0) / 1000.0
        
        print(f"\nθ = {theta:6.3f}, φ = {phi:6.3f}:")
        print(f"  P(|0⟩) = {prob_0:.4f}")
        print(f"  P(|1⟩) = {prob_1:.4f}")
    
    return parameterized_circuit


def expectation_value_example():
    """
    Demonstrate how to compute expectation values with CUDA-Q.
    
    We'll compute the expectation value of Pauli operators, which is
    fundamental for variational quantum algorithms.
    """
    print("\n" + "=" * 70)
    print("CUDA-Q: Expectation Values")
    print("=" * 70)
    
    @cudaq.kernel
    def circuit_for_x(theta: float, phi: float):
        """
        Circuit to measure σ_x.
        
        To measure σ_x, we apply Hadamard before measurement.
        """
        q = cudaq.qubit()
        ry(theta, q)
        rz(phi, q)
        h(q)  # Change basis to measure σ_x
        mz(q)
    
    @cudaq.kernel
    def circuit_for_z(theta: float, phi: float):
        """
        Circuit to measure σ_z (direct measurement).
        """
        q = cudaq.qubit()
        ry(theta, q)
        rz(phi, q)
        mz(q)  # Direct measurement in Z basis
    
    # Compute expectation values
    theta, phi = np.pi/4, np.pi/2
    
    print(f"\nComputing expectation values for θ = {theta:.3f}, φ = {phi:.3f}")
    
    # Measure σ_x
    result_x = cudaq.sample(circuit_for_x, theta, phi, shots_count=10000)
    prob_0_x = result_x.get('0', 0) / 10000.0
    prob_1_x = result_x.get('1', 0) / 10000.0
    expval_x = prob_0_x * 1 + prob_1_x * (-1)  # Eigenvalues: +1 for |0⟩, -1 for |1⟩
    
    # Measure σ_z
    result_z = cudaq.sample(circuit_for_z, theta, phi, shots_count=10000)
    prob_0_z = result_z.get('0', 0) / 10000.0
    prob_1_z = result_z.get('1', 0) / 10000.0
    expval_z = prob_0_z * 1 + prob_1_z * (-1)
    
    print(f"\n⟨σ_x⟩ = {expval_x:.6f}")
    print(f"⟨σ_z⟩ = {expval_z:.6f}")
    
    # Compare with analytical result
    # For |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)e^(iφ)|1⟩:
    # ⟨σ_x⟩ = sin(θ) cos(φ)
    expval_x_analytical = np.sin(theta) * np.cos(phi)
    print(f"\nAnalytical ⟨σ_x⟩ = {expval_x_analytical:.6f}")
    print(f"Error: {abs(expval_x - expval_x_analytical):.6f}")
    
    return expval_x, expval_z


def multi_qubit_example():
    """
    Demonstrate multi-qubit operations with CUDA-Q.
    
    We'll create an entangled state (Bell state) and measure it.
    """
    print("\n" + "=" * 70)
    print("CUDA-Q: Multi-Qubit Operations")
    print("=" * 70)
    
    @cudaq.kernel
    def bell_state():
        """
        Create a Bell state: (|00⟩ + |11⟩)/√2
        """
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        
        h(q0)  # Create superposition on first qubit
        x.ctrl(q0, q1)  # CNOT: controlled-NOT
        mz(q0)
        mz(q1)
    
    print("\nCreating Bell state...")
    result = cudaq.sample(bell_state, shots_count=10000)
    
    print("\nMeasurement results:")
    for state, count in result.items():
        prob = count / 10000.0
        print(f"  |{state}⟩: {prob:.4f} ({count} counts)")
    
    print("\nNote: Bell state should have equal probability for |00⟩ and |11⟩")
    print("      and zero probability for |01⟩ and |10⟩")
    
    return result


if __name__ == "__main__":
    print("\nWelcome to CUDA-Q!")
    print("CUDA-Q is NVIDIA's quantum computing platform for high-performance")
    print("quantum simulation and execution.\n")
    
    # Basic circuit
    basic_quantum_circuit()
    
    # Parameterized circuit
    parameterized_circuit()
    
    # Expectation values
    expectation_value_example()
    
    # Multi-qubit operations
    multi_qubit_example()
    
    print("\n" + "=" * 70)
    print("CUDA-Q basics demonstration complete!")
    print("=" * 70)

