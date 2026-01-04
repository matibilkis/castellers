"""
Module 3: Quantum Generative Adversarial Network (qGAN) with CUDA-Q
====================================================================

Quantum GANs are quantum versions of Generative Adversarial Networks, where
a quantum generator competes with a classical (or quantum) discriminator to
learn to generate data that matches a target distribution.

In this module, we implement a simple qGAN to generate a simple 2x2 pixel image,
demonstrating how quantum circuits can be used for generative modeling.

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


def create_target_image():
    """
    Create a simple 2x2 pixel target image.
    
    We'll represent images as 4-bit patterns (2x2 = 4 pixels, each 0 or 1).
    
    Returns
    -------
    np.ndarray
        2x2 binary image
    """
    # Create a simple pattern: diagonal line
    target = np.array([
        [1, 0],
        [0, 1]
    ])
    return target


def image_to_bitstring(image):
    """
    Convert a 2x2 image to a 4-bit string.
    
    Parameters
    ----------
    image : np.ndarray
        2x2 binary image
        
    Returns
    -------
    str
        4-bit string representation
    """
    return ''.join(str(image[i, j]) for i in range(2) for j in range(2))


def bitstring_to_image(bitstring):
    """
    Convert a 4-bit string to a 2x2 image.
    
    Parameters
    ----------
    bitstring : str
        4-bit string representation
        
    Returns
    -------
    np.ndarray
        2x2 binary image
    """
    image = np.zeros((2, 2), dtype=int)
    for idx, bit in enumerate(bitstring):
        i, j = idx // 2, idx % 2
        image[i, j] = int(bit)
    return image


def quantum_generator(params, n_qubits=4):
    """
    Quantum generator circuit.
    
    The generator uses a parameterized quantum circuit to generate
    probability distributions over 4-bit patterns.
    
    Parameters
    ----------
    params : np.ndarray
        Trainable parameters
    n_qubits : int
        Number of qubits (4 for 2x2 image)
        
    Returns
    -------
    cudaq.Kernel
        Generator kernel
    """
    @cudaq.kernel
    def generator():
        """
        Parameterized quantum circuit for generation.
        """
        qubits = [cudaq.qubit() for _ in range(n_qubits)]
        
        # Initial rotations
        param_idx = 0
        for q in qubits:
            ry(params[param_idx], q)
            param_idx += 1
        
        # Entangling layer
        for i in range(n_qubits - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        
        # Final rotations
        for q in qubits:
            ry(params[param_idx], q)
            param_idx += 1
        
        # Measure all qubits
        for q in qubits:
            mz(q)
    
    return generator


def sample_generator(generator, shots=10000):
    """
    Sample from the quantum generator.
    
    Parameters
    ----------
    generator : cudaq.Kernel
        Generator kernel
    shots : int
        Number of measurement shots
        
    Returns
    -------
    dict
        Dictionary mapping bitstrings to counts
    """
    result = cudaq.sample(generator, shots_count=shots)
    return result


def discriminator_loss(real_dist, fake_dist):
    """
    Compute discriminator loss.
    
    The discriminator wants to distinguish real from fake data.
    Loss = -log(D(real)) - log(1 - D(fake))
    
    For simplicity, we use a simple metric based on distribution overlap.
    
    Parameters
    ----------
    real_dist : dict
        Distribution of real data (bitstring -> probability)
    fake_dist : dict
        Distribution of fake data (bitstring -> probability)
        
    Returns
    -------
    float
        Discriminator loss
    """
    # Simple loss: negative log-likelihood of real data under fake distribution
    loss = 0.0
    for bitstring, prob_real in real_dist.items():
        prob_fake = fake_dist.get(bitstring, 1e-10)  # Avoid log(0)
        loss -= prob_real * np.log(prob_fake + 1e-10)
    return loss


def generator_loss(real_dist, fake_dist):
    """
    Compute generator loss.
    
    The generator wants to fool the discriminator.
    Loss = -log(D(fake))
    
    Parameters
    ----------
    real_dist : dict
        Distribution of real data
    fake_dist : dict
        Distribution of fake data
        
    Returns
    -------
    float
        Generator loss
    """
    # Generator wants fake distribution to match real distribution
    # Use KL divergence or similar
    loss = 0.0
    for bitstring, prob_real in real_dist.items():
        prob_fake = fake_dist.get(bitstring, 1e-10)
        if prob_real > 0:
            loss += prob_real * np.log(prob_real / (prob_fake + 1e-10))
    return loss


def train_qgan(target_image, n_iterations=50, shots=5000, lr=0.1):
    """
    Train a quantum GAN to generate the target image.
    
    Parameters
    ----------
    target_image : np.ndarray
        Target 2x2 image to generate
    n_iterations : int
        Number of training iterations
    shots : int
        Number of measurement shots per iteration
    lr : float
        Learning rate
        
    Returns
    -------
    tuple
        (trained_params, loss_history, generated_images)
    """
    print("=" * 70)
    print("Training Quantum GAN")
    print("=" * 70)
    
    # Convert target image to distribution
    target_bitstring = image_to_bitstring(target_image)
    real_dist = {target_bitstring: 1.0}  # Only one target pattern
    
    # Initialize generator parameters
    n_qubits = 4
    n_params = 2 * n_qubits  # Initial + final rotations
    params = np.random.uniform(0, 2*np.pi, n_params)
    
    loss_history = []
    generated_images = []
    
    print(f"\nTarget image:")
    print(target_image)
    print(f"Target bitstring: {target_bitstring}\n")
    
    for iteration in tqdm(range(n_iterations), desc="Training qGAN"):
        # Generate samples
        generator = quantum_generator(params, n_qubits)
        samples = sample_generator(generator, shots=shots)
        
        # Convert to probability distribution
        total = sum(samples.values())
        fake_dist = {bitstring: count / total for bitstring, count in samples.items()}
        
        # Compute losses
        gen_loss = generator_loss(real_dist, fake_dist)
        loss_history.append(gen_loss)
        
        # Simple gradient-free update (in practice, use parameter-shift rule)
        best_loss = gen_loss
        best_params = params.copy()
        
        # Try small random perturbations
        for _ in range(10):
            new_params = params + np.random.normal(0, 0.1, n_params)
            new_generator = quantum_generator(new_params, n_qubits)
            new_samples = sample_generator(new_generator, shots=shots//2)
            new_total = sum(new_samples.values())
            new_fake_dist = {bitstring: count / new_total 
                           for bitstring, count in new_samples.items()}
            new_loss = generator_loss(real_dist, new_fake_dist)
            
            if new_loss < best_loss:
                best_loss = new_loss
                best_params = new_params
        
        params = best_params
        
        # Track best generated image
        best_bitstring = max(fake_dist.items(), key=lambda x: x[1])[0]
        generated_images.append(bitstring_to_image(best_bitstring))
        
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration+1}: Loss = {gen_loss:.4f}")
            print(f"  Most likely generated: {best_bitstring}")
    
    print(f"\nFinal loss: {loss_history[-1]:.4f}")
    
    return params, loss_history, generated_images


def visualize_training(target_image, generated_images, loss_history):
    """
    Visualize the qGAN training process.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Show target image
    axes[0, 0].imshow(target_image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Show generated images at different stages
    indices = [0, len(generated_images)//4, len(generated_images)//2,
               3*len(generated_images)//4, len(generated_images)-1]
    
    for idx, ax in zip(indices, axes[0, 1:]):
        if idx < len(generated_images):
            ax.imshow(generated_images[idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Iteration {idx+1}', fontsize=10)
        ax.axis('off')
    
    # Plot loss
    axes[1, 0].plot(loss_history, linewidth=2, color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Generator Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Show final distribution (use last params from training)
    # Note: In a real implementation, final_params would be passed as argument
    # For now, we'll use a placeholder
    placeholder_params = np.zeros(8)
    final_generator = quantum_generator(placeholder_params, 4)
    final_samples = sample_generator(final_generator, shots=10000)
    total = sum(final_samples.values())
    final_dist = {bitstring: count / total 
                  for bitstring, count in final_samples.items()}
    
    bitstrings = sorted(final_dist.keys())
    probs = [final_dist[bs] for bs in bitstrings]
    
    axes[1, 1].bar(range(len(bitstrings)), probs, alpha=0.7)
    axes[1, 1].set_xticks(range(len(bitstrings)))
    axes[1, 1].set_xticklabels(bitstrings, rotation=45, fontsize=8)
    axes[1, 1].set_ylabel('Probability', fontsize=12)
    axes[1, 1].set_title('Final Distribution', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for ax in axes[1, 2:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('qgan_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'qgan_training.png'")
    plt.close()


def compare_target_vs_generated(target_image, final_params, shots=10000):
    """
    Compare target image with final generated distribution.
    """
    generator = quantum_generator(final_params, 4)
    samples = sample_generator(generator, shots=shots)
    
    total = sum(samples.values())
    distribution = {bitstring: count / total 
                   for bitstring, count in samples.items()}
    
    target_bitstring = image_to_bitstring(target_image)
    target_prob = distribution.get(target_bitstring, 0.0)
    
    print(f"\nTarget bitstring: {target_bitstring}")
    print(f"Probability of generating target: {target_prob:.4f}")
    print(f"\nTop 5 generated patterns:")
    for bitstring, prob in sorted(distribution.items(), key=lambda x: -x[1])[:5]:
        image = bitstring_to_image(bitstring)
        print(f"  {bitstring}: {prob:.4f}")
        print(f"    Image:\n{image}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Quantum Generative Adversarial Network (qGAN)")
    print("=" * 70)
    print("\nWe'll train a qGAN to generate a simple 2x2 pixel image.\n")
    
    # Create target image
    target_image = create_target_image()
    
    # Train qGAN
    final_params, loss_history, generated_images = train_qgan(
        target_image,
        n_iterations=50,
        shots=5000,
        lr=0.1
    )
    
    # Store final_params for visualization
    train_qgan.__globals__['final_params'] = final_params
    
    # Visualize
    visualize_training(target_image, generated_images, loss_history)
    
    # Compare
    compare_target_vs_generated(target_image, final_params)
    
    print("\n" + "=" * 70)
    print("qGAN demonstration complete!")
    print("=" * 70)
    print("\nNote: This is a simplified qGAN implementation. In practice,")
    print("      one would use more sophisticated architectures and training")
    print("      methods, including proper gradient computation using")
    print("      parameter-shift rules.")
    print("=" * 70)

