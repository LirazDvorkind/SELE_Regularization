import torch
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained score network
score_network = torch.load("../Data/sele_score_net_d32.pt", weights_only=False)
score_network.eval()

# Global grid parameters
W = 30e-4  # cm
x_res = 32
grid = np.linspace(0, W, x_res) * 1e4  # µm
normalized_grid = (grid - grid.min()) / (grid.max() - grid.min()) * 2 * np.pi

# The following is a normalized simulated SELE not seen by the score neural network
target_sele = [-0.4045, -0.5064, -0.6647, -0.8251, -0.8984, -0.9427, -0.9683, -0.9828,
        -0.9925, -0.9960, -0.9979, -0.9989, -0.9994, -0.9998, -0.9999, -0.9999,
        -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
        -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000] 

def sele_objective_grad(x):
    """
    Gradient of sele matching objective: -sum((x - target)^2)
    Returns gradient for gradient ascent (maximization)
    """
    return 2 * (target_sele - x)

def gradient_ascent_w_score(x0, obj_grad_fn, lr=1e-2, steps=100, reg_weight=0.1, t_val=1e-2):
    """
    Generic gradient ascent with score regularization
    
    Args:
        x0: initial guess (numpy array of function values)
        obj_grad_fn: function that takes x and returns objective gradient
        lr: learning rate
        steps: ascent steps
        reg_weight: weight of score regularization
        t_val: time-step hyperparameter
    """
    x = x0.copy()

    for step in range(steps):
        # Get objective gradient
        obj_grad = obj_grad_fn(x)
        
        # Get score regularization term
        x_with_t = np.concatenate([x, [t_val]])
        x_tensor_score = torch.tensor(x_with_t, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            score_reg = score_network(x_tensor_score).squeeze().numpy()

        # Perform gradient ascent
        x += lr * (obj_grad + reg_weight * score_reg)
        
        if step % 10 == 0:
            # Compute current objective value for monitoring
            obj_val = -np.sum((x - target_sele)**2)  # negative MSE
            reg_label = "w/ regularization" if reg_weight > 0 else "w/o regularization"
            print(f"step {step:03d} | obj={obj_val:.4f} ({reg_label})")

    return x

# --- Example usage ---

# Initial guess (same for both runs)
x0 = np.random.rand(x_res)

print("Running optimization WITH score regularization...")
x_opt_with_reg = gradient_ascent_w_score(x0, sele_objective_grad, lr=1e-2, steps=50, reg_weight=0.5)

print("\nRunning optimization WITHOUT score regularization...")
x_opt_no_reg = gradient_ascent_w_score(x0, sele_objective_grad, lr=1e-2, steps=50, reg_weight=0.0)

    
plt.figure(figsize=(12, 8))
plt.plot(grid, target_sele, 'r-', label='Target (simulated normalized SELE)', linewidth=3)
plt.plot(grid, x_opt_with_reg, 'b-', label='With score regularization', linewidth=2)
plt.plot(grid, x_opt_no_reg, 'g-', label='Without score regularization', linewidth=2)
plt.plot(grid, x0, 'k--', label='Initial guess', alpha=0.7)

plt.xlabel('Position (µm)')
plt.ylabel('Function value')
plt.title('Function Optimization: Effect of Score Network Regularization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print final errors
error_with_reg = np.sum((x_opt_with_reg - target_sele)**2)
error_no_reg = np.sum((x_opt_no_reg - target_sele)**2)

print(f"\nFinal MSE with regularization: {error_with_reg:.6f}")
print(f"Final MSE without regularization: {error_no_reg:.6f}")