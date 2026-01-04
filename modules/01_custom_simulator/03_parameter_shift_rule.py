"""
Module 2 - Part 1: Parameter-Shift Rule
=========================================

We explore how to compute gradients of quantum circuits using the
parameter-shift rule. This is a fundamental technique in quantum machine
learning, as it allows us to compute exact gradients (not approximations)
by evaluating the circuit at shifted parameter values.

Author: Matías Bilkis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from castellers import QuantumCircuit


def get_gradients(theta, phi, qc, shots=10):
    """
    Compute gradients using the parameter-shift rule.
    
    The parameter-shift rule states that for a rotation gate R(θ):
    ∂⟨H⟩/∂θ = (1/2) [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]
    
    This is exact (not an approximation) and holds for rotation gates.
    
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
    tuple
        (gradient w.r.t. theta, gradient w.r.t. phi)
    """
    # Gradient w.r.t. theta
    gtheta = 0
    for shift, constant in zip([np.pi/2, -np.pi/2], [1, -1]):
        circuit = [qc.ry(theta + shift), qc.rz(phi)]
        unitary = qc.unitary(circuit)
        output_state = qc.output_state(unitary)
        gtheta += constant * qc.observable_mean(output_state, operator="x", shots=shots)
    
    # Gradient w.r.t. phi
    gphi = 0
    for shift, constant in zip([np.pi/2, -np.pi/2], [1, -1]):
        circuit = [qc.ry(theta), qc.rz(phi + shift)]
        unitary = qc.unitary(circuit)
        output_state = qc.output_state(unitary)
        gphi += constant * qc.observable_mean(output_state, operator="x", shots=shots)
    
    return gtheta/2, gphi/2


def grad_th_analytical(theta, phi):
    """
    Analytical expression for the gradient w.r.t. theta.
    
    For ⟨H⟩ = sin(θ) cos(φ), we have:
    ∂⟨H⟩/∂θ = cos(θ) cos(φ)
    
    Parameters
    ----------
    theta : float or np.ndarray
        Rotation angle around y-axis
    phi : float or np.ndarray
        Rotation angle around z-axis
        
    Returns
    -------
    float or np.ndarray
        Gradient w.r.t. theta
    """
    return np.cos(theta) * np.cos(phi)


def verify_parameter_shift_rule():
    """
    Verify that the parameter-shift rule gives correct gradients.
    
    We compare the measured gradients (using parameter-shift rule) with
    the analytical gradients to confirm that our implementation is correct.
    """
    qc = QuantumCircuit()
    
    # Resolution for the parameter sweep
    ResTh, ResPh = 50, 50
    thetas_qc = np.linspace(0, np.pi, ResTh)
    phis_qc = np.linspace(0, 2*np.pi, ResPh)
    
    th1d, ph1d, grad_th1d = [], [], []
    
    print("Computing gradients using parameter-shift rule...")
    for th in tqdm(thetas_qc):
        for ph in phis_qc:
            grads = get_gradients(th, ph, qc, shots=25)
            th1d.append(th)
            ph1d.append(ph)
            grad_th1d.append(grads[0])
    
    # Analytical gradients
    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2*np.pi, 100)
    Thetas, Phis = np.meshgrid(thetas, phis)
    Grad_th = grad_th_analytical(Thetas, Phis)
    
    # Plot comparison
    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection='3d')
    
    # Measured gradients
    ax.scatter3D(th1d, ph1d, grad_th1d, c=grad_th1d, s=30, cmap='Blues',
                 label="Measured (parameter-shift)", alpha=0.6)
    
    # Analytical gradients
    im = ax.contour3D(Thetas, Phis, Grad_th, 50, vmin=-1, vmax=1,
                     cmap='Reds', alpha=0.5, label="Analytical")
    
    colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8)
    colorbar.set_label(r'$\partial \langle H \rangle / \partial \theta$', fontsize=14)
    
    ax.set_xticks(np.arange(0, np.pi + np.pi/4, np.pi/4))
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.set_yticks(np.arange(0, 2*np.pi + np.pi/2, np.pi/2))
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_zticks(np.arange(-1., 1.5, 0.5))
    
    ax.legend(fontsize=12)
    ax.set_zlim([-1, 1])
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\phi$', fontsize=14)
    ax.set_zlabel(r'$\partial \langle H \rangle / \partial \theta$', fontsize=14)
    ax.set_title("Gradient Landscape: Parameter-Shift Rule vs Analytical", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('gradient_landscape.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'gradient_landscape.png'")
    plt.close()
    
    # Compute error statistics
    th1d_arr = np.array(th1d)
    ph1d_arr = np.array(ph1d)
    grad_th1d_arr = np.array(grad_th1d)
    grad_th_analytical_arr = grad_th_analytical(th1d_arr, ph1d_arr)
    
    mse = np.mean((grad_th1d_arr - grad_th_analytical_arr)**2)
    print(f"\nMean squared error between measured and analytical gradients: {mse:.6f}")
    print("(Small error is expected due to finite shots)")


if __name__ == "__main__":
    print("=" * 70)
    print("Parameter-Shift Rule Demonstration")
    print("=" * 70)
    print("\nThe parameter-shift rule allows us to compute exact gradients")
    print("by evaluating the circuit at shifted parameter values:")
    print("  ∂⟨H⟩/∂θ = (1/2) [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]\n")
    
    verify_parameter_shift_rule()
    print("\nDone!")

