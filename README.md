# 🚀 Quantum Machine Learning Course

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-0.32+-purple.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Hands-on materials for learning Quantum Machine Learning from scratch to advanced algorithms**

[Features](#-features) • [Quick Start](#-quick-start) • [Modules](#-modules) • [Examples](#-examples)

</div>

---

## ✨ Features

- 🔬 **Custom Quantum Simulator**: Build a quantum circuit simulator from scratch to understand fundamentals
- 🧠 **PennyLane Integration**: Learn quantum machine learning with industry-standard tools
- ⚡ **CUDA-Q Examples**: High-performance quantum computing with NVIDIA's platform
- 📊 **Visual Learning**: Rich visualizations and plots for every concept
- 🎯 **Practical Examples**: Real-world applications including VQE, QAOA, and qGAN
- 🧪 **Fully Tested**: Comprehensive test suite with CI/CD pipeline

## 🎯 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/matibilkis/qml-course.git
cd qml-course

# Install dependencies
pip install -r requirements.txt

# Run your first quantum circuit
python modules/01_custom_simulator/01_basic_circuits.py
```

### Requirements

- Python 3.9+
- NumPy, Matplotlib, tqdm
- PennyLane (optional, for Module 2)
- CUDA-Q (optional, for Module 3)

## 📚 Modules

### Module 1: Custom Quantum Simulator 🏗️

**Learn quantum computing fundamentals by building from scratch**

Master the building blocks of quantum computing with our custom "Castellers" simulator. Understand gates, measurements, and quantum states at the lowest level.

```python
from castellers import QuantumCircuit

qc = QuantumCircuit()
circuit = [qc.ry(np.pi/2), qc.rz(np.pi)]
unitary = qc.unitary(circuit)
state = qc.output_state(unitary)
```

**Topics Covered:**
- ✅ Quantum gates and circuit composition
- ✅ State preparation and measurement
- ✅ Energy landscapes and optimization
- ✅ Parameter-shift rule for gradients
- ✅ Variational Quantum Algorithms (VQA)

**Files:**
- `01_basic_circuits.py` - Basic operations
- `02_energy_landscape.py` - Energy exploration
- `03_parameter_shift_rule.py` - Gradient computation
- `04_gradient_descent.py` - Complete VQA training

---

### Module 2: PennyLane Quantum ML 🧠

**Build quantum neural networks with modern tools**

Transition to professional quantum machine learning tools. Learn to build quantum neural networks, implement VQE, and leverage automatic differentiation.

```python
import pennylane as qml

@qml.qnode(dev)
def quantum_circuit(params):
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliX(0))
```

**Topics Covered:**
- ✅ PennyLane basics and automatic differentiation
- ✅ Variational Quantum Eigensolver (VQE)
- ✅ Entanglement and multi-qubit systems
- ✅ Quantum Neural Networks for classification
- ✅ Integration with classical ML frameworks

**Files:**
- `01_introduction.py` - PennyLane fundamentals
- `02_vqe_two_qubits.py` - VQE with entanglement
- `03_quantum_neural_network.py` - Complete QNN example

---

### Module 3: CUDA-Q Advanced Algorithms ⚡

**High-performance quantum computing with NVIDIA CUDA-Q**

Explore cutting-edge quantum algorithms on NVIDIA's platform. Implement QAOA for optimization and quantum GANs for generative modeling.

```python
import cudaq

@cudaq.kernel
def qaoa_ansatz(gamma, beta):
    q = cudaq.qubit()
    h(q)
    ry(gamma, q)
    rx(beta, q)
    mz(q)
```

**Topics Covered:**
- ✅ CUDA-Q platform fundamentals
- ✅ Variational Quantum Algorithms
- ✅ Quantum Approximate Optimization Algorithm (QAOA)
- ✅ Quantum Generative Adversarial Networks (qGAN)
- ✅ Combinatorial optimization applications

**Files:**
- `01_basics.py` - CUDA-Q fundamentals
- `02_vqa_training.py` - VQA implementation
- `03_qaoa.py` - QAOA for MaxCut
- `04_qgan.py` - Quantum GAN for image generation

## 🎨 Examples

### Example 1: Finding Ground States with VQE

```python
# Custom simulator approach
from modules.01_custom_simulator import optimize_vqa

params, energy = optimize_vqa(
    initial_params=[np.pi/4, np.pi/2],
    n_iterations=1000
)
```

### Example 2: Quantum Neural Network

```python
# PennyLane approach
from modules.02_pennylane.03_quantum_neural_network import train_qnn

qnn, n_params = create_qnn(n_layers=2)
params = train_qnn(qnn, n_params, X_train, y_train)
```

### Example 3: QAOA for Optimization

```python
# CUDA-Q approach
from modules.03_cudaq.03_qaoa import optimize_qaoa

gamma, beta, cost = optimize_qaoa(
    G, p=2, n_iterations=50
)
```

## 📊 Repository Structure

```
qml-course/
├── 📁 modules/
│   ├── 📁 01_custom_simulator/    # Custom quantum simulator
│   ├── 📁 02_pennylane/           # PennyLane examples
│   └── 📁 03_cudaq/               # CUDA-Q examples
├── 📁 tests/                       # Test suite
├── 📁 figures/                     # Visualizations
├── 📄 castellers.py               # Core simulator library
├── 📄 requirements.txt            # Dependencies
└── 📄 README.md                   # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov=castellers
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 Learning Path

1. **Start with Module 1** - Understand quantum fundamentals
2. **Move to Module 2** - Learn professional tools (PennyLane)
3. **Explore Module 3** - Advanced algorithms (CUDA-Q)
4. **Experiment** - Modify examples and create your own

## 🎓 Educational Context

This course material was originally developed for **Física Cuántica II** at the Autonomous University of Barcelona (May 2021) and has been expanded into a comprehensive QML course.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Matías Bilkis**

- GitHub: [@matibilkis](https://github.com/matibilkis)

## 🙏 Acknowledgments

- Original tutorial inspiration from Física Cuántica II course
- PennyLane team for excellent documentation
- NVIDIA for CUDA-Q platform
- Quantum computing community

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ for the quantum computing community

</div>
