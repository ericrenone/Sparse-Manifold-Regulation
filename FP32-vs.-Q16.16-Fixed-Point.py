import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Configuration
# ----------------------------
tasks = ["LM", "Image Class", "Tabular"]
x = np.arange(len(tasks))
width = 0.35

# Data dictionary to make the plotting logic "DRY" (Don't Repeat Yourself)
# Format: { 'Metric Name': (FP32_Data, Q16_Data, Y_Label, Annotation_Text, Offset) }
bench_data = {
    "Time to Convergence": (
        np.array([100, 80, 60]), 
        np.array([101, 81, 61]), 
        "Time (s)", "Almost identical", 5
    ),
    "Final Task Error": (
        np.array([0.05, 0.10, 0.08]), 
        np.array([0.055, 0.105, 0.085]), 
        "Error Rate", "Small ~0.5% increase", 0.01
    ),
    "Energy Usage": (
        np.array([100, 80, 60]), 
        np.array([15, 12, 9]), 
        "Rel. Units", "5-10x lower", 5
    ),
    "Wasted Computation": (
        np.array([42, 42, 42]), 
        np.array([32, 32, 32]), 
        "Waste (%)", "Reduced hardware waste", 3
    )
}

# ----------------------------
# 2. Plotting Logic
# ----------------------------
plt.style.use('seaborn-v0_8-muted') # Cleaner, modern aesthetic
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("FP32 vs Q16.16 SOTA Fixed-Point Hardware Benchmark", fontsize=18, weight='bold')

axes_flat = axes.flatten()

for i, (title, (fp_val, q_val, ylabel, note, offset)) in enumerate(bench_data.items()):
    ax = axes_flat[i]
    
    # Plot bars
    ax.bar(x - width/2, fp_val, width, label="FP32 H100", color="#2c3e50", alpha=0.8)
    ax.bar(x + width/2, q_val, width, label="Q16.16 SOTA", color="#e67e22", alpha=0.9)
    
    # Styling
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel(ylabel, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(frameon=True)
    
    # Dynamic Annotation
    for j in range(len(tasks)):
        ax.annotate(
            note, 
            xy=(x[j] + width/2, q_val[j]), 
            xytext=(0, 10), # 10pts offset above the bar
            textcoords="offset points",
            ha='center',
            arrowprops=dict(facecolor='black', arrowstyle="->", alpha=0.5),
            fontsize=8,
            fontstyle='italic'
        )

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
