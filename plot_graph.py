import re
import matplotlib.pyplot as plt
import pandas as pd

# File mapping
files = {
    'Ablation A': 'ablation_a.log',
    'Ablation B': 'ablation_b.log',
    'Ablation C': 'ablation_c.log',
    'Ablation D': 'ablation_d.log'
}

# Regex patterns for parsing training logs and validation logs
train_pattern = re.compile(
    r"iteration\s+(\d+)/\s*50000\s+\|.*?lm loss:\s+([0-9\.E\+\-]+)(?:.*?capacity_pricing_loss:\s+([0-9\.E\+\-]+))?(?:.*?load_balancing_loss:\s+([0-9\.E\+\-]+))?")
val_pattern = re.compile(
    r"validation loss at iteration (\d+) \| lm loss value: ([0-9\.E\+\-]+) \| lm loss PPL: ([0-9\.E\+\-]+)")

data = {name: {'iter': [], 'lm_loss': [], 'cap_price_loss': [], 'val_iter': [], 'val_ppl': []} for name in files}

# Parse logs
for name, filepath in files.items():
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Match training iteration stats
                t_match = train_pattern.search(line)
                if t_match:
                    iteration = int(t_match.group(1))
                    lm_loss = float(t_match.group(2))

                    # Capture capacity loss or load balancing loss based on ablation
                    aux_loss = 0.0
                    if t_match.group(3):  # Capacity price loss
                        aux_loss = float(t_match.group(3))
                    elif t_match.group(4):  # Load balancing loss
                        aux_loss = float(t_match.group(4))

                    data[name]['iter'].append(iteration)
                    data[name]['lm_loss'].append(lm_loss)
                    data[name]['cap_price_loss'].append(aux_loss)

                # Match validation stats
                v_match = val_pattern.search(line)
                if v_match:
                    iteration = int(v_match.group(1))
                    val_ppl = float(v_match.group(3))
                    data[name]['val_iter'].append(iteration)
                    data[name]['val_ppl'].append(val_ppl)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Ensure the logs are in the same directory.")

# Plotting the Convergence
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = {'Ablation A': 'blue', 'Ablation B': 'orange', 'Ablation C': 'green', 'Ablation D': 'red'}

for name in files:
    if data[name]['iter']:
        # 1. LM Loss Convergence
        axes[0].plot(data[name]['iter'], data[name]['lm_loss'], label=name, color=colors[name], alpha=0.7)

        # 2. Auxiliary Loss Convergence (Capacity Price / Load Balancing)
        # Skip plotting zeros for Ablation A which has no auxiliary loss tracked here
        valid_aux = [loss for loss in data[name]['cap_price_loss'] if loss > 0]
        if valid_aux:
            if "B" in name:
                axes[1].plot(data[name]['iter'][:len(valid_aux)], valid_aux, label=f"{name} (Aux Loss)", color=colors[name],
                         alpha=0.7)
            else:
                axes[1].plot(data[name]['iter'][:len(valid_aux)], valid_aux, label=f"{name} (Capacity Price Loss)",
                             color=colors[name],
                             alpha=0.7)

        # 3. Validation Perplexity Convergence
        axes[2].plot(data[name]['val_iter'], data[name]['val_ppl'], label=name, color=colors[name], marker='o',
                     markersize=3)

# Formatting charts
axes[0].set_title('Training LM Loss Convergence')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('LM Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].set_title('Expert Balancing Loss')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Balancing Loss')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

axes[2].set_title('Validation Perplexity (PPL)')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Perplexity')
axes[2].set_yscale('log')  # Log scale is often better for PPL visualization
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('convergence_metrics.png', dpi=300, bbox_inches='tight')
print("Successfully generated and saved 'convergence_metrics.png' to your directory.")