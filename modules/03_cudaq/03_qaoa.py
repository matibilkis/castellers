"""
Module 3: Quantum Approximate Optimization Algorithm (QAOA) with CUDA-Q
========================================================================

QAOA is a variational quantum algorithm designed for combinatorial optimization
problems. It uses a specific ansatz structure with alternating "problem" and
"mixer" Hamiltonians.

In this module, we implement a basic QAOA example for the MaxCut problem,
one of the most common applications of QAOA.

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
import networkx as nx


def create_maxcut_problem(n_nodes=4):
    """
    Create a simple MaxCut problem instance.
    
    MaxCut: Given a graph, find a partition of vertices that maximizes
    the number of edges between the two partitions.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph
        
    Returns
    -------
    networkx.Graph
        Graph representing the MaxCut problem
    """
    # Create a simple graph (ring or complete graph)
    G = nx.Graph()
    
    # Create a ring graph (each node connected to 2 neighbors)
    for i in range(n_nodes):
        G.add_edge(i, (i + 1) % n_nodes)
    
    # Add some additional edges for a more interesting problem
    if n_nodes >= 4:
        G.add_edge(0, 2)
    
    return G


def maxcut_cost_function(G, bitstring):
    """
    Compute the cost (number of cut edges) for a given partition.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the problem
    bitstring : str
        Binary string representing the partition (0 or 1 for each node)
        
    Returns
    -------
    int
        Number of edges cut by this partition
    """
    cost = 0
    for edge in G.edges():
        i, j = edge
        if bitstring[i] != bitstring[j]:
            cost += 1
    return cost


def qaoa_ansatz(G, gamma, beta, p=1):
    """
    QAOA ansatz for MaxCut problem.
    
    The QAOA ansatz alternates between:
    1. Problem Hamiltonian (phase operator): exp(-i γ C)
    2. Mixer Hamiltonian (X rotations): exp(-i β X)
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the MaxCut problem
    gamma : float or list
        Problem parameters (one per layer)
    beta : float or list
        Mixer parameters (one per layer)
    p : int
        Number of QAOA layers
        
    Returns
    -------
    cudaq.Kernel
        QAOA ansatz kernel
    """
    if isinstance(gamma, (int, float)):
        gamma = [gamma] * p
    if isinstance(beta, (int, float)):
        beta = [beta] * p
    
    @cudaq.kernel
    def ansatz():
        """
        QAOA ansatz for MaxCut.
        """
        # Initialize qubits in superposition
        qubits = [cudaq.qubit() for _ in range(len(G.nodes()))]
        for q in qubits:
            h(q)
        
        # Apply p layers
        for layer in range(p):
            # Problem Hamiltonian: exp(-i γ C)
            # For MaxCut, C = Σ_{edges} (1 - Z_i Z_j)/2
            for edge in G.edges():
                i, j = edge
                # Apply exp(-i γ Z_i Z_j) using CNOT and RZ
                x.ctrl(qubits[i], qubits[j])
                rz(2 * gamma[layer], qubits[j])
                x.ctrl(qubits[i], qubits[j])
            
            # Mixer Hamiltonian: exp(-i β X)
            for q in qubits:
                rx(2 * beta[layer], q)
        
        # Measure all qubits
        for q in qubits:
            mz(q)
    
    return ansatz


def qaoa_expectation_value(G, gamma, beta, p=1, shots=10000):
    """
    Compute the expectation value of the cost function using QAOA.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the MaxCut problem
    gamma : float or list
        Problem parameters
    beta : float or list
        Mixer parameters
    p : int
        Number of QAOA layers
    shots : int
        Number of measurement shots
        
    Returns
    -------
    float
        Expectation value of the cost function
    """
    ansatz = qaoa_ansatz(G, gamma, beta, p)
    result = cudaq.sample(ansatz, shots_count=shots)
    
    # Compute expected cost
    total_shots = sum(result.values())
    expected_cost = 0.0
    
    for bitstring, count in result.items():
        cost = maxcut_cost_function(G, bitstring)
        prob = count / total_shots
        expected_cost += prob * cost
    
    return expected_cost


def optimize_qaoa(G, p=1, n_iterations=50, learning_rate=0.1, shots=5000):
    """
    Optimize QAOA parameters to maximize the expected cut value.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the MaxCut problem
    p : int
        Number of QAOA layers
    n_iterations : int
        Number of optimization steps
    learning_rate : float
        Learning rate for gradient descent
    shots : int
        Number of measurement shots
        
    Returns
    -------
    tuple
        (optimized_gamma, optimized_beta, cost_history)
    """
    print("=" * 70)
    print(f"Optimizing QAOA for MaxCut (p={p} layers)")
    print("=" * 70)
    
    # Initialize parameters
    gamma = np.random.uniform(0, 2*np.pi, p)
    beta = np.random.uniform(0, np.pi, p)
    
    cost_history = []
    
    print(f"\nGraph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Initial parameters:")
    print(f"  γ = {gamma}")
    print(f"  β = {beta}\n")
    
    for iteration in tqdm(range(n_iterations), desc="Optimizing QAOA"):
        # Compute cost
        cost = qaoa_expectation_value(G, gamma, beta, p, shots=shots)
        cost_history.append(cost)
        
        # Simple gradient-free optimization (random search with improvement)
        # In practice, one would use parameter-shift rule or other methods
        best_cost = cost
        best_gamma = gamma.copy()
        best_beta = beta.copy()
        
        # Try small random perturbations
        for _ in range(10):
            new_gamma = gamma + np.random.normal(0, 0.1, p)
            new_beta = beta + np.random.normal(0, 0.1, p)
            new_cost = qaoa_expectation_value(G, new_gamma, new_beta, p, shots=shots//2)
            
            if new_cost > best_cost:
                best_cost = new_cost
                best_gamma = new_gamma
                best_beta = new_beta
        
        gamma = best_gamma
        beta = best_beta
    
    print(f"\nOptimized parameters:")
    print(f"  γ = {gamma}")
    print(f"  β = {beta}")
    print(f"Final expected cost: {cost_history[-1]:.4f}")
    
    return gamma, beta, cost_history


def find_best_cut(G, gamma, beta, p=1, shots=10000):
    """
    Find the best cut by sampling from the optimized QAOA state.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the MaxCut problem
    gamma : float or list
        Optimized problem parameters
    beta : float or list
        Optimized mixer parameters
    p : int
        Number of QAOA layers
    shots : int
        Number of measurement shots
        
    Returns
    -------
    tuple
        (best_bitstring, best_cost, all_results)
    """
    ansatz = qaoa_ansatz(G, gamma, beta, p)
    result = cudaq.sample(ansatz, shots_count=shots)
    
    # Find best cut
    best_bitstring = None
    best_cost = -1
    
    for bitstring, count in result.items():
        cost = maxcut_cost_function(G, bitstring)
        if cost > best_cost:
            best_cost = cost
            best_bitstring = bitstring
    
    return best_bitstring, best_cost, result


def visualize_maxcut_solution(G, bitstring):
    """
    Visualize the MaxCut solution.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph representing the problem
    bitstring : str
        Binary string representing the partition
    """
    plt.figure(figsize=(10, 8))
    
    # Color nodes based on partition
    node_colors = ['red' if bit == '0' else 'blue' for bit in bitstring]
    
    # Draw graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    
    # Highlight cut edges
    cut_edges = []
    for edge in G.edges():
        i, j = edge
        if bitstring[i] != bitstring[j]:
            cut_edges.append(edge)
    
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='green',
                          width=3, alpha=0.7, label='Cut edges')
    
    plt.title(f'MaxCut Solution (Cost = {maxcut_cost_function(G, bitstring)})',
              fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('qaoa_maxcut_solution.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'qaoa_maxcut_solution.png'")
    plt.close()


def plot_optimization_history(cost_history):
    """
    Plot the QAOA optimization history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=3, color='blue', alpha=0.7)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Expected Cost", fontsize=14)
    plt.title("QAOA Optimization History", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('qaoa_optimization.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'qaoa_optimization.png'")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Quantum Approximate Optimization Algorithm (QAOA)")
    print("=" * 70)
    print("\nQAOA is a variational quantum algorithm for combinatorial")
    print("optimization problems. We'll apply it to the MaxCut problem.\n")
    
    # Create problem instance
    G = create_maxcut_problem(n_nodes=4)
    print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Optimize QAOA
    gamma, beta, cost_history = optimize_qaoa(
        G, p=1, n_iterations=30, learning_rate=0.1, shots=3000
    )
    
    # Find best solution
    best_bitstring, best_cost, results = find_best_cut(G, gamma, beta, p=1)
    
    print(f"\nBest solution found: {best_bitstring}")
    print(f"Cost: {best_cost}")
    print(f"\nDistribution of solutions:")
    for bitstring, count in sorted(results.items(), key=lambda x: -x[1])[:5]:
        cost = maxcut_cost_function(G, bitstring)
        print(f"  {bitstring}: cost={cost}, count={count} ({count/sum(results.values())*100:.1f}%)")
    
    # Visualize
    plot_optimization_history(cost_history)
    visualize_maxcut_solution(G, best_bitstring)
    
    print("\n" + "=" * 70)
    print("QAOA demonstration complete!")
    print("=" * 70)

