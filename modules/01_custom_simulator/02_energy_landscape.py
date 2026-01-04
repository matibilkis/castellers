"""
Module 2 - Part 1: Energy Landscape Exploration
================================================

We explore the energy landscape for different parameter choices.
This helps us understand how the number of measurement shots affects
the accuracy of our estimates, and allows us to numerically find the
optimal parameters that minimize the energy.

Author: Matías Bilkis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from castellers import QuantumCircuit


def mean_value_x(theta, phi, qc, shots=10):
    """
    Compute the mean value of σ_x for given parameters.
    
    This function generates a circuit determined by (θ, φ) and estimates
    the mean value of σ_x by measuring in the z-basis with a given
    number of shots.
    
    Parameters
    ----------
    theta : float
        Rotation angle around y-axis
    phi : float
        Rotation angle around z-axis
    qc : QuantumCircuit
        Quantum circuit instance
    shots : int or np.inf
        Number of measurement shots
        
    Returns
    -------
    float
        Expectation value ⟨σ_x⟩
    """
    circuit = [qc.ry(theta), qc.rz(phi)]
    unitary = qc.unitary(circuit)
    output_state = qc.output_state(unitary)
    return qc.observable_mean(output_state, operator="x", shots=shots)


def get_phis(theta, qc, shots=np.inf):
    """
    Sweep over different values of φ (z-axis rotation) for a fixed θ.
    
    Parameters
    ----------
    theta : float
        Fixed rotation angle around y-axis
    qc : QuantumCircuit
        Quantum circuit instance
    shots : int or np.inf
        Number of measurement shots
        
    Returns
    -------
    list
        List of expectation values for different φ values
    """
    mphi = []
    phis = np.linspace(0, 2*np.pi, 100)
    for ph in phis:
        mphi.append(mean_value_x(theta, ph, qc, shots=shots))
    return mphi


def plot_energy_vs_shots():
    """
    Visualize how the number of shots affects the accuracy of energy estimates.
    
    We plot the expectation value ⟨H⟩ = ⟨σ_x⟩ for different parameter choices
    and different numbers of measurement shots.
    """
    qc = QuantumCircuit()
    phis = np.linspace(0, 2*np.pi, 100)
    
    plt.figure(figsize=(20, 6))
    
    for k, shots in enumerate([10, 100, np.inf]):
        ax = plt.subplot(1, 3, k+1)
        
        fun_str = str(shots) if isinstance(shots, int) else "infinite"
        ax.set_title(f"Number of shots = {fun_str}", fontsize=14)
        
        for theta, color, label in zip(
            [0, np.pi/4, np.pi/2],
            ["green", "blue", "red"],
            [r'$\theta = 0$', r'$\theta = \frac{\pi}{4}$', r'$\theta = \frac{\pi}{2}$']
        ):
            energies = get_phis(theta, qc, shots=shots)
            ax.plot(phis, energies, color=color, linewidth=3, alpha=0.8, label=label)
        
        ax.set_xticks(np.arange(0, 2*np.pi + np.pi/2, np.pi/2))
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.legend(fontsize=12)
        ax.set_xlabel(r'$\phi$', fontsize=14)
        ax.set_ylabel(r'$\langle H \rangle(\theta, \phi)$', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_vs_shots.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'energy_vs_shots.png'")
    plt.close()


def energy_landscape_analytical(theta, phi):
    """
    Analytical expression for the energy landscape.
    
    For the Hamiltonian H = σ_x and the state:
    |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2) e^(iφ)|1⟩
    
    we have: ⟨ψ|H|ψ⟩ = sin(θ) cos(φ)
    
    Parameters
    ----------
    theta : float or np.ndarray
        Rotation angle around y-axis
    phi : float or np.ndarray
        Rotation angle around z-axis
        
    Returns
    -------
    float or np.ndarray
        Energy expectation value
    """
    return np.sin(theta) * np.cos(phi)


def plot_3d_energy_landscape():
    """
    Create a 3D visualization of the energy landscape.
    """
    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2*np.pi, 100)
    
    Thetas, Phis = np.meshgrid(thetas, phis)
    Energies = energy_landscape_analytical(Thetas, Phis)
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    im = ax.contour3D(Thetas, Phis, Energies, 50, cmap="viridis", vmin=-1, vmax=1)
    colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8)
    colorbar.set_label(r'$\langle H \rangle$', fontsize=14)
    
    ax.set_xticks(np.arange(0, np.pi + np.pi/4, np.pi/4))
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.set_yticks(np.arange(0, 2*np.pi + np.pi/2, np.pi/2))
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_zticks(np.arange(-1., 1.5, 0.5))
    
    ax.set_zlim([-1, 1])
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\phi$', fontsize=14)
    ax.set_zlabel(r'$\langle H \rangle$', fontsize=14)
    ax.set_title("Energy Landscape", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('energy_landscape_3d.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'energy_landscape_3d.png'")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Energy Landscape Exploration")
    print("=" * 70)
    print("\nNote: The optimal parameters (minimum energy) are θ = π/2 and φ = π")
    print("This corresponds to the state |ψ⟩ = (|0⟩ - |1⟩)/√2 = |-⟩\n")
    
    print("Plotting energy vs shots...")
    plot_energy_vs_shots()
    
    print("\nPlotting 3D energy landscape...")
    plot_3d_energy_landscape()
    
    print("\nDone!")

