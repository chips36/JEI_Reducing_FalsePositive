import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Common settings and colors
thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Colors are defined globally so they are easy to change
COLOR_YOLO = '#2ca02c'    # Green
COLOR_NUDENET = '#ff7f0e' # Orange

# ==========================================
# [Data - YOLO] (300 images per class, 10 male images not detected)
# ==========================================
y_f_total, y_m_total = 300, 300
y_m_tn = np.array([281, 279, 274, 269, 143])
y_m_fp = np.array([9, 11, 16, 21, 147])
y_f_tp = np.array([211, 237, 269, 293, 299])
y_f_fn = np.array([89, 63, 31, 7, 1])

y_m_base = 143 # YOLO male baseline
y_f_base = 299 # YOLO female baseline

# ==========================================
# [Data - NUDENET] (300 images per class)
# ==========================================
n_f_total, n_m_total = 300, 300
n_m_tn = np.array([261, 256, 252, 250, 133])
n_m_fp = np.array([7, 12, 16, 18, 135])
n_f_tp = np.array([183, 198, 223, 254, 279])
n_f_fn = np.array([99, 84, 59, 28, 3])

n_m_base = 133 # NudeNet male baseline
n_f_base = 279 # NudeNet female baseline

def save_plot(filename):
    """Save the current figure at 300 dpi and display it."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"📈 저장 완료: {filename}")
    plt.show()

# ---------------------------------------------------------
# 1. Male dataset comparison
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(thresholds, y_m_tn, marker='o', color=COLOR_YOLO, linewidth=3, label='YOLO + Face (Miss: 10)')
plt.plot(thresholds, n_m_tn, marker='s', color=COLOR_NUDENET, linewidth=3, label='NudeNet + Face (Miss: 32)')

# Baselines use the same color as their pipeline to avoid confusion
plt.axhline(y=y_m_base, color=COLOR_YOLO, linestyle='--', alpha=0.5, label='YOLO Baseline')
plt.axhline(y=n_m_base, color=COLOR_NUDENET, linestyle='--', alpha=0.5, label='NudeNet Baseline')

plt.title('Male Dataset: Correct Count (Total 300)', fontweight='bold')
plt.ylabel('Correct Count (TN)')
plt.legend(frameon=True, shadow=True)
save_plot('1_male_comparison_fixed_colors.png')

# ---------------------------------------------------------
# 2. Female dataset comparison
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(thresholds, y_f_tp, marker='o', color=COLOR_YOLO, linewidth=3, label='YOLO + Face (Miss: 0)')
plt.plot(thresholds, n_f_tp, marker='s', color=COLOR_NUDENET, linewidth=3, label='NudeNet + Face (Miss: 18)')

plt.axhline(y=y_f_base, color=COLOR_YOLO, linestyle='--', alpha=0.5, label='YOLO Baseline')
plt.axhline(y=n_f_base, color=COLOR_NUDENET, linestyle='--', alpha=0.5, label='NudeNet Baseline')

plt.title('Female Dataset: Correct Count (Total 300)', fontweight='bold')
plt.ylabel('Correct Count (TP)')
plt.legend(frameon=True, shadow=True)
save_plot('2_female_comparison_fixed_colors.png')

# ---------------------------------------------------------
# 3. Overall system accuracy comparison
# ---------------------------------------------------------
y_total_acc = (y_f_tp + y_m_tn) / (y_f_total + y_m_total)
n_total_acc = (n_f_tp + n_m_tn) / (n_f_total + n_m_total)

plt.figure(figsize=(10, 6.5))
# YOLO in thick green, NudeNet in thick orange
plt.plot(thresholds, y_total_acc, marker='D', color=COLOR_YOLO, linewidth=4, markersize=10, label='YOLO')
plt.plot(thresholds, n_total_acc, marker='X', color=COLOR_NUDENET, linewidth=4, markersize=10, label='NudeNet')

plt.title('Overall System Accuracy Comparison', fontweight='bold', fontsize=16)
plt.xlabel('Confidence Threshold')
plt.ylabel('Accuracy Score')
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right', frameon=True, shadow=True)
save_plot('3_overall_accuracy_fixed_colors.png')
