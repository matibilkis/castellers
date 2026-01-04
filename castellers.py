"""
Castellers: A custom quantum simulator for educational purposes

This module implements a simple quantum circuit simulator from scratch, designed
to help understand the fundamental operations of quantum computing "under the hood".
Named after the Catalan tradition of building human castles, this code builds
quantum circuits layer by layer, from the ground up.

Author: Matías Bilkis
"""

import numpy as np


class QuantumCircuit:
    """
    A simple quantum circuit simulator for 1-qubit systems.
    
    This class provides the basic building blocks for quantum circuits:
    - Quantum gates (rotations, Hadamard, phase gates)
    - State preparation and evolution
    - Measurement operations
    - Observable expectation values
    
    The implementation is intentionally simple to facilitate understanding
    of the underlying quantum mechanics principles.
    """
    
    def __init__(self, n_qubits=1):
        """
        Initialize a quantum circuit.
        
        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits in the circuit (default: 1)
            Note: Currently only 1-qubit operations are fully implemented
        """
        self.n_qubits = n_qubits
        if n_qubits > 1:
            raise NotImplementedError(
                "Multi-qubit operations are not yet fully implemented. "
                "For multi-qubit circuits, please use PennyLane or other quantum libraries."
            )

    def rz(self, theta):
        """
        Rotation around the z-axis.
        
        This gate applies a phase rotation: Rz(θ)|0⟩ = |0⟩, Rz(θ)|1⟩ = e^(-iθ)|1⟩
        
        Note: We use a convention where Rz(θ)|0⟩ = |0⟩ (no phase on |0⟩).
        This is equivalent to applying Exp[-i θ σ_z/2] with an additional phase.
        
        Parameters
        ----------
        theta : float
            Rotation angle in radians
            
        Returns
        -------
        np.ndarray
            2x2 unitary matrix representing the rotation
        """
        return np.array([[1, 0], [0, np.exp(-theta * 1j)]])
        # Alternative (standard) convention:
        # return np.array([[np.exp(theta*0.5j), 0], [0, np.exp(-theta*0.5j)]])

    def ry(self, theta):
        """
        Rotation around the y-axis.
        
        This gate performs a rotation on the Bloch sphere around the y-axis:
        Ry(θ) = exp(-i θ σ_y/2) = cos(θ/2) I - i sin(θ/2) σ_y
        
        Parameters
        ----------
        theta : float
            Rotation angle in radians
            
        Returns
        -------
        np.ndarray
            2x2 unitary matrix representing the rotation
        """
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])

    @property
    def H(self):
        """
        Hadamard gate.
        
        The Hadamard gate creates superpositions: H|0⟩ = (|0⟩ + |1⟩)/√2
        
        Returns
        -------
        np.ndarray
            2x2 Hadamard matrix
        """
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @property
    def S(self):
        """
        Phase gate (S gate).
        
        Applies a π/2 phase: S|0⟩ = |0⟩, S|1⟩ = i|1⟩
        
        Returns
        -------
        np.ndarray
            2x2 phase gate matrix
        """
        return np.array([[1, 0], [0, 1j]])

    def unitary(self, list_of_gates):
        """
        Construct the unitary matrix representing a sequence of gates.
        
        The gates are applied in the order they appear in the list (left to right
        in circuit notation, which means right to left in matrix multiplication).
        
        Parameters
        ----------
        list_of_gates : list of np.ndarray
            List of gate matrices to be applied in sequence
            
        Returns
        -------
        np.ndarray
            Combined unitary matrix representing the entire circuit
        """
        if not list_of_gates:
            return np.eye(2)  # Identity if no gates
        
        unitary = list_of_gates[-1]
        # Apply gates in reverse order (circuit reads left-to-right, 
        # but matrix multiplication is right-to-left)
        for gate in list_of_gates[::-1][1:]:
            unitary = np.matmul(unitary, gate)
        return unitary

    def output_state(self, unitary):
        """
        Compute the output state after applying a unitary to the fiducial state |0⟩.
        
        Parameters
        ----------
        unitary : np.ndarray
            Unitary matrix representing the circuit
            
        Returns
        -------
        np.ndarray
            Quantum state vector (in computational basis)
        """
        # Initial state is |0⟩ = [1, 0]^T
        initial_state = np.array([1, 0])
        return np.matmul(unitary, initial_state)

    def get_probability(self, projector, output_state, shots=np.inf):
        """
        Compute the probability of measuring a state in a given projector.
        
        For finite shots, this simulates the probabilistic nature of quantum
        measurements by sampling from a binomial distribution.
        
        Parameters
        ----------
        projector : np.ndarray
            Projector operator (e.g., |0⟩⟨0| or |1⟩⟨1|)
        output_state : np.ndarray
            Quantum state vector
        shots : int or np.inf, optional
            Number of measurement shots. If np.inf, returns exact probability.
            (default: np.inf)
            
        Returns
        -------
        float
            Probability (or estimated probability for finite shots)
            
        Notes
        -----
        This implementation only works for 1-qubit systems. For multi-qubit
        systems, one needs to sample from a multinomial distribution.
        """
        if shots == np.inf:
            # Exact probability: |⟨projector|state⟩|²
            return np.abs(np.matmul(projector, output_state))**2
        else:
            # Validate inputs
            assert shots > 0, "Number of shots must be positive"
            assert isinstance(shots, int), "Number of shots must be an integer"
            assert self.n_qubits == 1, "Finite shots only implemented for 1 qubit"
            
            # Compute exact probability
            pr_exact = np.abs(np.matmul(projector, output_state))**2
            
            # Simulate measurements: sample from binomial distribution
            # This mimics what happens in a real quantum computer
            estimated_probability = np.random.binomial(shots, pr_exact) / shots
            return estimated_probability

    def observable_mean(self, output_state, operator="x", shots=np.inf):
        """
        Compute the expectation value of a Pauli observable.
        
        To measure Pauli operators, we need to perform a change of basis:
        - σ_z: measure directly in computational basis
        - σ_x: apply Hadamard, then measure in computational basis
        - σ_y: apply S† then Hadamard, then measure in computational basis
        
        Parameters
        ----------
        output_state : np.ndarray
            Quantum state vector
        operator : str, optional
            Pauli operator to measure: "x", "y", or "z" (default: "x")
        shots : int or np.inf, optional
            Number of measurement shots (default: np.inf)
            
        Returns
        -------
        float
            Expectation value ⟨operator⟩
            
        Notes
        -----
        This implementation only works for 1-qubit systems. For multi-qubit
        systems, the change-of-basis operations become more complex.
        See: https://docs.microsoft.com/en-us/azure/quantum/concepts-pauli-measurements
        """
        # Projectors onto computational basis states
        projectors = np.eye(2)
        
        # Change of basis for different Pauli operators
        if operator.lower() == "x":
            # To measure σ_x, apply Hadamard to rotate to eigenbasis
            output_state = np.matmul(self.H, output_state)
        elif operator.lower() == "y":
            # To measure σ_y, apply S† then Hadamard
            output_state = np.matmul(np.conjugate(self.S), output_state)
            output_state = np.matmul(self.H, output_state)
        elif operator.lower() == "z":
            # σ_z is already diagonal in computational basis
            pass
        else:
            raise ValueError(f"Unknown operator: {operator}. Use 'x', 'y', or 'z'")
        
        # Compute expectation value: Σ eigenvalue_i × probability_i
        # For Pauli operators: eigenvalues are +1 (|0⟩) and -1 (|1⟩)
        avg = 0
        eigenvalues = [1, -1]
        for projector, eigenval in zip(projectors, eigenvalues):
            prob = self.get_probability(projector, output_state, shots=shots)
            avg += eigenval * prob
        
        return avg
