"""
Module 2 - Part 1: Custom Quantum Simulator
============================================

This module demonstrates the fundamental operations of quantum circuits
using our custom simulator "Castellers". We start with a simple 1-qubit
problem to understand the building blocks of quantum computing.

Author: Matías Bilkis
"""

import numpy as np
import sys
import os

# Add parent directory to path to import castellers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from castellers import QuantumCircuit


def demonstrate_basic_circuit():
    """
    Demonstrate basic quantum circuit operations with Castellers.
    
    We work with a 1-qubit (spin 1/2 system) Hamiltonian H = σ_x = X.
    We consider a simple parametrized quantum circuit (PQC), composed of
    two rotations around the y-axis and z-axis respectively (with angles θ, φ).
    
    The circuit: |0⟩ ---- Ry(θ) ---- Rz(φ) ----
    
    The result of the unitary transformation applied to |0⟩ is:
    |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2) e^(iφ)|1⟩
    
    with θ ∈ [0, π] and φ ∈ [0, 2π).
    """
    print("=" * 70)
    print("Basic Quantum Circuit Demonstration")
    print("=" * 70)
    
    qc = QuantumCircuit()
    
    # Example: θ = π, φ = 0
    phi = 0
    theta = np.pi
    
    circuit = [qc.ry(theta), qc.rz(phi)]
    print("\nCircuit: |0⟩ ---- Ry(θ) ---- Rz(φ) ----")
    print(f"Parameters: θ = {theta:.3f}, φ = {phi:.3f}\n")
    
    unitary = qc.unitary(circuit)
    print("Unitary transformation representing the circuit:")
    print(np.round(unitary, 3))
    print()
    
    output_state = qc.output_state(unitary)
    print("State at the end of the circuit (initial state was |0⟩):")
    print(output_state)
    print()
    
    # Measure different observables
    avg_sigma_x = qc.observable_mean(output_state, operator="x")
    avg_sigma_y = qc.observable_mean(output_state, operator="y")
    avg_sigma_z = qc.observable_mean(output_state, operator="z")
    
    print("Expectation values:")
    print(f"  ⟨σ_x⟩ = {avg_sigma_x:.6f}")
    print(f"  ⟨σ_y⟩ = {avg_sigma_y:.6f}")
    print(f"  ⟨σ_z⟩ = {avg_sigma_z:.6f}")
    print()
    
    return qc, output_state


if __name__ == "__main__":
    demonstrate_basic_circuit()

