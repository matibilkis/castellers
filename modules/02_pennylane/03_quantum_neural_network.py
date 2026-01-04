"""
Module 2 - Part 2: Quantum Neural Networks with PennyLane
===========================================================

We now build a complete Quantum Neural Network (QNN) example using PennyLane.
This demonstrates how quantum circuits can be used as machine learning models,
with automatic differentiation and integration with classical optimizers.

We'll create a simple QNN for a classification task, showing the full workflow
from data preparation to training and evaluation.

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


def create_data(n_samples=100):
    """
    Create a simple 2D classification dataset.
    
    We'll use a circular pattern: points inside a circle belong to class 0,
    points outside belong to class 1.
    
    Parameters
    ----------
    n_samples : int
        Number of data points
        
    Returns
    -------
    tuple
        (X, y) where X is the data and y are the labels
    """
    np.random.seed(42)
    
    # Generate random points in [-1, 1] x [-1, 1]
    X = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Labels: 0 if inside circle (radius 0.6), 1 otherwise
    y = ((X[:, 0]**2 + X[:, 1]**2) > 0.6**2).astype(int)
    
    return X, y


def data_reuploading_ansatz(params, x, n_layers=2):
    """
    Data re-uploading ansatz: a popular architecture for quantum neural networks.
    
    The idea is to encode classical data into the quantum circuit multiple times,
    interleaved with parameterized rotations. This allows the quantum circuit
    to learn complex functions of the input data.
    
    Parameters
    ----------
    params : array-like
        Trainable parameters
    x : array-like
        Input data point [x₀, x₁]
    n_layers : int
        Number of layers in the ansatz
    """
    n_qubits = 2
    n_params_per_layer = 3 * n_qubits  # 3 rotations per qubit
    
    for layer in range(n_layers):
        # Encode data
        for qubit in range(n_qubits):
            qml.RY(x[qubit], wires=qubit)
        
        # Parameterized rotations
        param_idx = layer * n_params_per_layer
        for qubit in range(n_qubits):
            qml.RY(params[param_idx + 3*qubit + 0], wires=qubit)
            qml.RZ(params[param_idx + 3*qubit + 1], wires=qubit)
            qml.RY(params[param_idx + 3*qubit + 2], wires=qubit)
        
        # Entangling layer
        if layer < n_layers - 1:  # No CNOT after last layer
            qml.CNOT(wires=[0, 1])


def create_qnn(n_layers=2):
    """
    Create a Quantum Neural Network using PennyLane.
    
    Parameters
    ----------
    n_layers : int
        Number of layers in the ansatz
        
    Returns
    -------
    tuple
        (qnode, n_params) where qnode is the quantum circuit and n_params is
        the number of trainable parameters
    """
    n_qubits = 2
    n_params_per_layer = 3 * n_qubits
    n_params = n_layers * n_params_per_layer
    
    dev = qml.device('default.qubit', wires=n_qubits, shots=None)
    
    @qml.qnode(dev, interface='autograd')
    def qnn(params, x):
        """
        Quantum neural network circuit.
        
        Parameters
        ----------
        params : array-like
            Trainable parameters
        x : array-like
            Input data point
            
        Returns
        -------
        float
            Expectation value of PauliZ on qubit 0 (used as output)
        """
        data_reuploading_ansatz(params, x, n_layers=n_layers)
        return qml.expval(qml.PauliZ(0))
    
    return qnn, n_params


def predict(qnn, params, x):
    """
    Make a prediction using the QNN.
    
    Parameters
    ----------
    qnn : callable
        Quantum neural network function
    params : array-like
        Trained parameters
    x : array-like
        Input data point
        
    Returns
    -------
    int
        Predicted class (0 or 1)
    """
    output = qnn(params, x)
    # Map output from [-1, 1] to [0, 1]
    return 1 if output > 0 else 0


def accuracy(qnn, params, X, y):
    """
    Compute classification accuracy.
    
    Parameters
    ----------
    qnn : callable
        Quantum neural network function
    params : array-like
        Trained parameters
    X : array-like
        Input data
    y : array-like
        True labels
        
    Returns
    -------
    float
        Accuracy (between 0 and 1)
    """
    predictions = [predict(qnn, params, x) for x in X]
    return np.mean(predictions == y)


def loss_function(qnn, params, X, y):
    """
    Loss function for binary classification.
    
    We use a simple squared loss: (output - target)²
    
    Parameters
    ----------
    qnn : callable
        Quantum neural network function
    params : array-like
        Trainable parameters
    X : array-like
        Input data
    y : array-like
        True labels (0 or 1)
        
    Returns
    -------
    float
        Loss value
    """
    loss = 0.0
    for x, target in zip(X, y):
        output = qnn(params, x)
        # Map target from {0, 1} to {-1, 1}
        target_mapped = 2 * target - 1
        loss += (output - target_mapped)**2
    return loss / len(X)


def train_qnn(qnn, n_params, X_train, y_train, n_iterations=100, lr=0.1):
    """
    Train the Quantum Neural Network.
    
    Parameters
    ----------
    qnn : callable
        Quantum neural network function
    n_params : int
        Number of trainable parameters
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    n_iterations : int
        Number of training iterations
    lr : float
        Learning rate
        
    Returns
    -------
    tuple
        (trained_params, loss_history, accuracy_history)
    """
    print("=" * 70)
    print("Training Quantum Neural Network")
    print("=" * 70)
    
    # Initialize parameters randomly
    np.random.seed(42)
    params = np.random.uniform(0, 2*np.pi, n_params)
    
    # Optimizer
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    
    loss_history = []
    accuracy_history = []
    
    print(f"\nInitial accuracy: {accuracy(qnn, params, X_train, y_train):.4f}")
    print(f"Initial loss: {loss_function(qnn, params, X_train, y_train):.6f}\n")
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Compute loss and gradients
        loss = loss_function(qnn, params, X_train, y_train)
        grad_loss = qml.grad(loss_function, argnum=1)
        grads = grad_loss(qnn, params, X_train, y_train)
        
        # Update parameters
        params = opt.step(lambda p: loss_function(qnn, p, X_train, y_train), params)
        
        # Track progress
        loss_history.append(loss)
        acc = accuracy(qnn, params, X_train, y_train)
        accuracy_history.append(acc)
        
        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration+1:3d}: Loss = {loss:.6f}, Accuracy = {acc:.4f}")
    
    print(f"\nFinal accuracy: {accuracy_history[-1]:.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    return params, loss_history, accuracy_history


def plot_training_results(loss_history, accuracy_history):
    """
    Plot training curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(loss_history, linewidth=3, color='red', alpha=0.7)
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.set_title("Training Loss", fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracy_history, linewidth=3, color='blue', alpha=0.7)
    ax2.set_xlabel("Iteration", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.set_title("Training Accuracy", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('qnn_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'qnn_training.png'")
    plt.close()


def plot_decision_boundary(qnn, params, X, y):
    """
    Visualize the decision boundary learned by the QNN.
    """
    # Create a grid of points
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Predict for each point in the grid
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_point = np.array([xx[i, j], yy[i, j]])
            Z[i, j] = qnn(params, x_point)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='QNN Output')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=50, linewidths=1)
    plt.colorbar(scatter, label='True Label')
    
    plt.xlabel('x₀', fontsize=14)
    plt.ylabel('x₁', fontsize=14)
    plt.title('QNN Decision Boundary', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qnn_decision_boundary.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'qnn_decision_boundary.png'")
    plt.close()


if __name__ == "__main__":
    # Create data
    print("Creating dataset...")
    X, y = create_data(n_samples=100)
    
    # Create QNN
    print("\nCreating Quantum Neural Network...")
    qnn, n_params = create_qnn(n_layers=2)
    print(f"Number of parameters: {n_params}")
    
    # Train QNN
    params, loss_history, accuracy_history = train_qnn(
        qnn, n_params, X, y,
        n_iterations=100,
        lr=0.1
    )
    
    # Plot results
    plot_training_results(loss_history, accuracy_history)
    plot_decision_boundary(qnn, params, X, y)
    
    print("\n" + "=" * 70)
    print("Quantum Neural Network training complete!")
    print("=" * 70)

