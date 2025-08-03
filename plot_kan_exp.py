import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python plot_kan.py <input_folder> <output_folder>")
    print("Example: python plot_kan.py /home/allen/ws/logkan /home/allen/ws/mamba_quad")
    sys.exit(1)

input_folder = sys.argv[1]
output_folder = sys.argv[2]

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Only plot specific experiments
import glob
import matplotlib.cm as cm

def find_all_experiments(base_folder):
    """Find all experiment log files in subfolders"""
    experiments = []
    colors = cm.tab10(np.linspace(0, 1, 10))  # Generate distinct colors
    
    # Find all subdirectories
    subdirs = [d for d in os.listdir(base_folder) 
               if os.path.isdir(os.path.join(base_folder, d))]
    subdirs.sort()  # Sort for consistent ordering
    
    color_idx = 0
    for subdir in subdirs:
        # Look for log.csv files in the subfolder structure
        possible_paths = [
            os.path.join(base_folder, subdir, 'A1MoveGround/0/log.csv'),
            os.path.join(base_folder, subdir, '*/A1MoveGround/0/log.csv'),
            os.path.join(base_folder, subdir, 'thin-vision/A1MoveGround/0/log.csv'),
            os.path.join(base_folder, subdir, 'thin/A1MoveGround/0/log.csv')
        ]
        
        log_file = None
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                log_file = matches[0]
                break
        
        if log_file and os.path.exists(log_file):
            experiments.append((subdir, log_file, colors[color_idx % len(colors)]))
            color_idx += 1
            print(f"Found experiment: {subdir} -> {log_file}")
    
    return experiments

# Read the experiment data from input folder
experiments = find_all_experiments(input_folder)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('KAN Experiments (First 610 Epochs)', fontsize=16, fontweight='bold')

# Plot 1: Rewards over time
ax1 = axes[0, 0]
for name, path, color in experiments:
    try:
        df = pd.read_csv(path)
        # Limit to first 610 epochs
        df = df[df['EPOCH'] <= 610]
        samples = df['Total Frames']
        rewards = df['Running_Average_Rewards'].dropna()
        samples = samples[:len(rewards)]
        
        # Plot mean line
        ax1.plot(samples, rewards, label=f'{name}', color=color, linewidth=2)
        
        # Add standard deviation bands (using rolling std)
        if len(rewards) > 1:
            rolling_std = rewards.rolling(window=5, center=True).std()
            ax1.fill_between(samples, rewards - rolling_std, rewards + rolling_std, 
                           alpha=0.2, color=color)
    except Exception as e:
        print(f"Error plotting {name}: {e}")

ax1.set_xlabel('Sample Number')
ax1.set_ylabel('Running Average Rewards')
ax1.set_title('Reward Learning Curves')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Value Function Loss
ax2 = axes[0, 1]
for name, path, color in experiments:
    try:
        df = pd.read_csv(path)
        # Limit to first 610 epochs
        df = df[df['EPOCH'] <= 610]
        samples = df['Total Frames']
        vf_loss = df['Training/vf_loss_Mean'].dropna()
        vf_std = df['Training/vf_loss_Std'].dropna()
        samples = samples[:len(vf_loss)]
        
        ax2.plot(samples, vf_loss, label=f'{name}', color=color, linewidth=2)
        ax2.fill_between(samples, vf_loss - vf_std, vf_loss + vf_std, 
                        alpha=0.2, color=color)
    except Exception as e:
        print(f"Error plotting VF loss {name}: {e}")

ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Value Function Loss')
ax2.set_title('Value Function Loss with Std Dev')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Advantages Mean
ax3 = axes[1, 0]
for name, path, color in experiments:
    try:
        df = pd.read_csv(path)
        # Limit to first 610 epochs
        df = df[df['EPOCH'] <= 610]
        samples = df['Total Frames']
        advs_mean = df['advs/mean_Mean'].dropna()
        advs_std = df['advs/mean_Std'].dropna()
        samples = samples[:len(advs_mean)]
        
        ax3.plot(samples, advs_mean, label=f'{name}', color=color, linewidth=2)
        ax3.fill_between(samples, advs_mean - advs_std, advs_mean + advs_std, 
                        alpha=0.2, color=color)
    except Exception as e:
        print(f"Error plotting advantages {name}: {e}")

ax3.set_xlabel('Sample Number')
ax3.set_ylabel('Advantages Mean')
ax3.set_title('Advantages Mean with Std Dev')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 4: Learning Stability (Coefficient of Variation)
ax4 = axes[1, 1]
stability_data = []
for name, path, color in experiments:
    try:
        df = pd.read_csv(path)
        # Limit to first 610 epochs
        df = df[df['EPOCH'] <= 610]
        rewards = df['Running_Average_Rewards'].dropna()
        vf_loss = df['Training/vf_loss_Mean'].dropna()
        advs_mean = df['advs/mean_Mean'].dropna()
        
        # Calculate CV for each metric
        reward_cv = (rewards.std() / abs(rewards.mean())) * 100 if rewards.mean() != 0 else 0
        vf_cv = (vf_loss.std() / abs(vf_loss.mean())) * 100 if vf_loss.mean() != 0 else 0
        advs_cv = (advs_mean.std() / abs(advs_mean.mean())) * 100 if advs_mean.mean() != 0 else 0
        
        stability_data.append({
            'Experiment': name,
            'Rewards CV': reward_cv,
            'VF Loss CV': vf_cv,
            'Advantages CV': min(advs_cv, 250)  # Cap for visualization
        })
    except Exception as e:
        print(f"Error calculating stability {name}: {e}")

if stability_data:
    exp_names = [d['Experiment'] for d in stability_data]
    reward_cvs = [d['Rewards CV'] for d in stability_data]
    vf_cvs = [d['VF Loss CV'] for d in stability_data]
    adv_cvs = [d['Advantages CV'] for d in stability_data]
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    ax4.bar(x - width, reward_cvs, width, label='Rewards CV', alpha=0.8)
    ax4.bar(x, vf_cvs, width, label='VF Loss CV', alpha=0.8)
    ax4.bar(x + width, adv_cvs, width, label='Advantages CV', alpha=0.8)
    
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Learning Stability (Lower = More Stable)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.replace(' ', '\n') for name in exp_names], fontsize=8)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(output_folder, 'learning_curves_kan_exp.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Learning curves saved as '{output_file}'")

# Print analysis
print(f"\n=== LEARNING CURVE DEVIATIONS (First 610 Epochs) - {input_folder} ===")
for name, path, color in experiments:
    try:
        df = pd.read_csv(path)
        # Limit to first 610 epochs
        df = df[df['EPOCH'] <= 610]
        rewards = df['Running_Average_Rewards'].dropna()
        vf_loss = df['Training/vf_loss_Mean'].dropna()
        advs_mean = df['advs/mean_Mean'].dropna()
        
        print(f'\n--- {name} ({len(rewards)} data points, epochs 0-610) ---')
        if len(rewards) > 0:
            print(f'Reward std: {rewards.std():.2f} (CV: {(rewards.std()/abs(rewards.mean()))*100:.1f}%)')
        if len(vf_loss) > 0:
            print(f'VF Loss std: {vf_loss.std():.4f} (CV: {(vf_loss.std()/abs(vf_loss.mean()))*100:.1f}%)')
        if len(advs_mean) > 0:
            print(f'Advs/mean std: {advs_mean.std():.4f} (CV: {(advs_mean.std()/abs(advs_mean.mean()))*100:.1f}%)')
        
        # Convergence analysis
        if len(rewards) > 5:
            final_portion = max(1, len(rewards)//5)
            final_rewards = rewards[-final_portion:]
            print(f'Final {final_portion} data points reward std: {final_rewards.std():.2f}')
        
    except Exception as e:
        print(f'{name}: Error - {e}')