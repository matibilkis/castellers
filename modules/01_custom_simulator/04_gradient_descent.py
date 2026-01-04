"""
Module 2 - Part 1: Gradient Descent Optimization
===================================================

We implement gradient descent to optimize the parameters of our quantum
circuit. This demonstrates how variational quantum algorithms work:
1. Prepare a quantum state with some parameters
2. Measure the cost function (energy)
3. Compute gradients using parameter-shift rule
4. Update parameters using gradient descent
5. Repeat until convergence

Author: Matías Bilkis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from castellers import QuantumCircuit

# Import helper functions with relative imports
import importlib.util
spec_energy = importlib.util.spec_from_file_location(
    "energy_landscape", 
    os.path.join(os.path.dirname(__file__), "02_energy_landscape.py")
)
energy_module = importlib.util.module_from_spec(spec_energy)
spec_energy.loader.exec_module(energy_module)
mean_value_x = energy_module.mean_value_x
energy_landscape_analytical = energy_module.energy_landscape_analytical

spec_grad = importlib.util.spec_from_file_location(
    "parameter_shift_rule",
    os.path.join(os.path.dirname(__file__), "03_parameter_shift_rule.py")
)
grad_module = importlib.util.module_from_spec(spec_grad)
spec_grad.loader.exec_module(grad_module)
get_gradients = grad_module.get_gradients


def gradient_descent_step(params, grads, lr=0.1):
    """
    Perform one step of gradient descent.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameter values [theta, phi]
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


def optimize_vqa(initial_params, n_iterations=1000, learning_rate=0.01, shots=100):
    """
    Optimize a variational quantum algorithm using gradient descent.
    
    This function implements the full VQA loop:
    1. Prepare state with current parameters
    2. Measure energy
    3. Compute gradients using parameter-shift rule
    4. Update parameters
    5. Repeat
    
    Parameters
    ----------
    initial_params : np.ndarray
        Initial parameter values [theta, phi]
    n_iterations : int
        Number of optimization steps
    learning_rate : float
        Learning rate for gradient descent
    shots : int or np.inf
        Number of measurement shots
        
    Returns
    -------
    tuple
        (trajectory_params, trajectory_energy, final_params)
    """
    qc = QuantumCircuit()
    params = initial_params.copy()
    trajectory_params = [params.copy()]
    trajectory_energy = []
    
    # Initial energy
    energy = mean_value_x(params[0], params[1], qc, shots=shots)
    trajectory_energy.append(energy)
    
    print(f"Initial parameters: θ = {params[0]:.4f}, φ = {params[1]:.4f}")
    print(f"Initial energy: {energy:.6f}\n")
    
    for iteration in tqdm(range(n_iterations), desc="Optimizing"):
        # Compute gradients using parameter-shift rule
        grads = get_gradients(params[0], params[1], qc, shots=shots)
        
        # Update parameters
        params = gradient_descent_step(params, np.array(grads), lr=learning_rate)
        
        # Measure energy (use infinite shots for accurate tracking)
        energy = mean_value_x(params[0], params[1], qc, shots=np.inf)
        
        trajectory_params.append(params.copy())
        trajectory_energy.append(energy)
    
    print(f"\nFinal parameters: θ = {params[0]:.4f}, φ = {params[1]:.4f}")
    print(f"Final energy: {trajectory_energy[-1]:.6f}")
    print(f"Optimal energy (analytical): {energy_landscape_analytical(np.pi/2, np.pi):.6f}")
    
    return np.array(trajectory_params), np.array(trajectory_energy), params


def plot_optimization_trajectory(trajectory_params, trajectory_energy):
    """
    Visualize the optimization trajectory in parameter space and energy.
    """
    # 2D plot: Energy vs iterations
    plt.figure(figsize=(20, 8))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(trajectory_energy, linewidth=3, color="red", alpha=0.7, label="Castellers training")
    ax1.axhline(y=-1, color='black', linestyle='--', linewidth=2, label="Optimal energy")
    ax1.set_xlabel("Gradient Descent Steps", fontsize=16)
    ax1.set_ylabel("Cost function (energy)", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Energy Convergence", fontsize=18)
    
    # 3D plot: Trajectory in parameter space
    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2*np.pi, 100)
    Thetas, Phis = np.meshgrid(thetas, phis)
    Energies = energy_landscape_analytical(Thetas, Phis)
    
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    
    # Energy landscape
    im = ax2.contour3D(Thetas, Phis, Energies, 20, cmap="binary", vmin=-1, vmax=1, alpha=0.3)
    
    # Optimization trajectory
    ax2.plot3D(trajectory_params[:, 0], trajectory_params[:, 1], trajectory_energy,
               alpha=1, linewidth=4, color="green", label="Trajectory")
    
    # Initial and final points
    ax2.scatter3D(trajectory_params[0, 0], trajectory_params[0, 1], trajectory_energy[0],
                  s=300, color="red", label="Initial point", marker='o')
    ax2.scatter3D(trajectory_params[-1, 0], trajectory_params[-1, 1], trajectory_energy[-1],
                  s=300, color="blue", label="Final point", marker='*')
    
    ax2.set_xticks(np.arange(0, np.pi + np.pi/4, np.pi/4))
    ax2.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax2.set_yticks(np.arange(0, 2*np.pi + np.pi/2, np.pi/2))
    ax2.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax2.set_zticks(np.arange(-1., 1.5, 0.5))
    
    ax2.legend(fontsize=12)
    ax2.set_zlim([-1, 1])
    ax2.set_xlabel(r'$\theta$', fontsize=14)
    ax2.set_ylabel(r'$\phi$', fontsize=14)
    ax2.set_zlabel(r'$\langle H \rangle$', fontsize=14)
    ax2.set_title("Optimization Trajectory", fontsize=18)
    
    plt.tight_layout()
    plt.savefig('optimization_trajectory.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'optimization_trajectory.png'")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Gradient Descent Optimization")
    print("=" * 70)
    print("\nWe will optimize the parameters of our quantum circuit to find")
    print("the ground state of H = σ_x using gradient descent.\n")
    
    # Initialize parameters (slightly off from optimal to see convergence)
    initial_params = np.array([np.pi/4, np.pi/2 + 0.1])
    
    # Run optimization
    trajectory_params, trajectory_energy, final_params = optimize_vqa(
        initial_params,
        n_iterations=1000,
        learning_rate=0.01,
        shots=100
    )
    
    # Visualize results
    plot_optimization_trajectory(trajectory_params, trajectory_energy)
    
    print("\nDone!")

