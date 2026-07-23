import numpy as np
import matplotlib.pyplot as plt


def generate_task(n_samples, x1_mean, x1_std, noise_scale=0.1):
    """Generates data from a fixed SCM, allowing shifts in the root cause."""
    # 1. Root Cause (shifted between tasks)
    x1 = np.random.normal(x1_mean, x1_std, n_samples)

    # 2. X2 is caused by X1 (Fixed non-linear mechanism)
    x2 = np.sin(x1 * 2) + np.random.normal(0, noise_scale, n_samples)

    # 3. Target y is caused by X1 and X2 (Fixed mechanism)
    y = 2.5 * x1 - 1.5 * x2 + np.random.normal(0, noise_scale, n_samples)

    return x1, x2, y


# --- Meta-SCM Generation ---
np.random.seed(42)

# Task A: Large dataset, base distribution
A_x1, A_x2, A_y = generate_task(n_samples=1000, x1_mean=0.0, x1_std=1.0)

# Task B: Small dataset, severe covariate shift (shifted mean, tighter variance)
B_x1, B_x2, B_y = generate_task(n_samples=50, x1_mean=2.5, x1_std=0.3)

# --- Plotting the Relatedness ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Feature Interaction (X1 -> X2)
ax1.scatter(A_x1, A_x2, alpha=0.3, label='Task A (Large, Base)', color='blue')
ax1.scatter(B_x1, B_x2, alpha=1.0, label='Task B (Small, Shifted)', color='red', edgecolor='black')
ax1.set_title('Feature Interaction: X1 vs X2\n(Notice Task B lives on Task A\'s manifold)')
ax1.set_xlabel('Feature X1')
ax1.set_ylabel('Feature X2')
ax1.legend()

# Plot 2: Target Mechanism (X2 -> y)
ax2.scatter(A_x2, A_y, alpha=0.3, label='Task A', color='blue')
ax2.scatter(B_x2, B_y, alpha=1.0, label='Task B', color='red', edgecolor='black')
ax2.set_title('Target Mechanism: X2 vs y\n(Tasks share the same causal rules)')
ax2.set_xlabel('Feature X2')
ax2.set_ylabel('Target y')
ax2.legend()

plt.tight_layout()
plt.show()