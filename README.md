# Quantum Machine Learning Course - Hands-On Materials

This repository contains hands-on materials for a Quantum Machine Learning (QML) course, focusing on practical implementations using custom simulators, PennyLane, and CUDA-Q.

**Author:** Matías Bilkis

## Course Structure

The course is organized into three main modules:

### Module 1: Custom Quantum Simulator (`modules/01_custom_simulator/`)

Learn quantum computing fundamentals by building a quantum simulator from scratch. This module uses "Castellers", a custom quantum circuit simulator that helps understand what happens "under the hood".

**Contents:**
- `01_basic_circuits.py`: Basic quantum circuit operations
- `02_energy_landscape.py`: Exploring energy landscapes and measurement statistics
- `03_parameter_shift_rule.py`: Understanding gradient computation in quantum circuits
- `04_gradient_descent.py`: Complete VQA training loop with gradient descent

### Module 2: PennyLane Quantum Machine Learning (`modules/02_pennylane/`)

Introduction to PennyLane, a powerful quantum machine learning library. Learn to build quantum neural networks and variational quantum algorithms using modern quantum software.

**Contents:**
- `01_introduction.py`: PennyLane basics and automatic differentiation
- `02_vqe_two_qubits.py`: Variational Quantum Eigensolver for 2-qubit systems
- `03_quantum_neural_network.py`: Complete QNN implementation for classification

### Module 3: CUDA-Q Quantum Computing (`modules/03_cudaq/`)

Explore NVIDIA's CUDA-Q platform for high-performance quantum computing. Implement advanced algorithms including QAOA and quantum GANs.

**Contents:**
- `01_basics.py`: CUDA-Q fundamentals and basic operations
- `02_vqa_training.py`: Variational Quantum Algorithm training with CUDA-Q
- `03_qaoa.py`: Quantum Approximate Optimization Algorithm for MaxCut
- `04_qgan.py`: Quantum Generative Adversarial Network for image generation

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- tqdm

### Optional Dependencies

**PennyLane:**
```bash
pip install pennylane
```

**CUDA-Q:**
Follow the installation instructions at: https://nvidia.github.io/cuda-quantum/latest/install.html

## Usage

Each module can be run independently. For example:

```bash
# Run custom simulator examples
python modules/01_custom_simulator/01_basic_circuits.py

# Run PennyLane examples
python modules/02_pennylane/01_introduction.py

# Run CUDA-Q examples
python modules/03_cudaq/01_basics.py
```

## Core Library

The `castellers.py` file contains the core quantum circuit simulator class, which is used throughout Module 1. This custom implementation helps understand the fundamental operations of quantum computing.

## Figures

The `figures/` directory contains visualizations used in the course materials.

## License

This material is part of a course on Quantum Machine Learning. Please refer to the course documentation for usage terms.

## Acknowledgments

This course material is based on the original tutorial given at "Física Quàntica II", Autonomous University of Barcelona, in May 2021.
