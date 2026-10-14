import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Data: YOLOv8 + proposed method across gender confidence thresholds
thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]

# [Female data] N=300
f_tp = np.array([211, 237, 269, 293, 299])
f_fn = np.array([89, 63, 31, 7, 1])
f_total = 300
f_base_correct = 299

# [Male data] N=290 (detected images only)
m_tn = np.array([281, 279, 274, 269, 143])
m_fp = np.array([9, 11, 16, 21, 147])
m_total = 290
m_base_correct = 143

# Plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

def save_plot(filename):
    """Save the current figure at 300 dpi and display it."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"저장 완료: {filename}")
    plt.show()

# ---------------------------------------------------------
# 1-3. Correct counts vs. threshold (proposed method vs. baseline)
# ---------------------------------------------------------
for i, (data, base, title, color, name) in enumerate([
    (m_tn, m_base_correct, 'Male Dataset: Correct Count', '#2ca02c', '1_male_count'),
    (f_tp, f_base_correct, 'Female Dataset: Correct Count', '#1f77b4', '2_female_count'),
    (f_tp + m_tn, f_base_correct + m_base_correct, 'Combined Dataset: Correct Count', '#9467bd', '3_combined_count')
]):
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, data, marker='o', color=color, linewidth=3.5, markersize=10, label='Proposed Method')
    plt.axhline(y=base, color='#d62728', linestyle='--', linewidth=2.5, label='Baseline')
    plt.title(title, fontweight='bold', fontsize=16)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Count')
    plt.legend(frameon=True, shadow=True)
    save_plot(f'{name}.png')

# ---------------------------------------------------------
# 4. Metrics (male class) - green palette
# ---------------------------------------------------------
m_prec = m_tn / (m_tn + f_fn)
m_rec = m_tn / m_total
m_f1 = 2 * (m_prec * m_rec) / (m_prec + m_rec)
m_acc = m_tn / m_total

plt.figure(figsize=(9, 6))
plt.plot(thresholds, m_acc, marker='s', color='#2ca02c', linewidth=3, label='Accuracy (Male)')
plt.plot(thresholds, m_f1, marker='x', color='#00441b', linestyle='--', linewidth=2.5, label='F1-score (Male)')
plt.title('Male Dataset Performance', fontweight='bold', fontsize=16)
plt.ylim(0.4, 1.05)
plt.legend()
save_plot('4_male_metrics.png')

# ---------------------------------------------------------
# 5. Metrics (female class) - blue palette
# ---------------------------------------------------------
f_prec = f_tp / (f_tp + m_fp)
f_rec = f_tp / f_total
f_f1 = 2 * (f_prec * f_rec) / (f_prec + f_rec)
f_acc = f_tp / f_total

plt.figure(figsize=(9, 6))
plt.plot(thresholds, f_acc, marker='o', color='#1f77b4', linewidth=3, label='Accuracy (Female)')
plt.plot(thresholds, f_f1, marker='x', color='#08306b', linestyle='--', linewidth=2.5, label='F1-score (Female)')
plt.title('Female Dataset Performance', fontweight='bold', fontsize=16)
plt.ylim(0.4, 1.05)
plt.legend()
save_plot('5_female_metrics.png')

# ---------------------------------------------------------
# 6. Metrics (combined) - high-contrast black & orange
# ---------------------------------------------------------
total_acc = (f_tp + m_tn) / (f_total + m_total)
total_f1 = (f_f1 + m_f1) / 2  # Macro-averaged F1 over the two classes

plt.figure(figsize=(10, 6.5))
# Accuracy in solid black
plt.plot(thresholds, total_acc, marker='D', color='#000000', linewidth=4, markersize=12, label='Overall Accuracy')
# F1-score in bright orange
plt.plot(thresholds, total_f1, marker='X', color='#ff7f0e', linestyle='--', linewidth=3, markersize=12, label='Overall F1-score')

plt.title('Combined Dataset Performance', fontweight='bold', fontsize=18)
plt.xlabel('Confidence Threshold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.ylim(0.6, 1.0) # Emphasize the main range of change
plt.xticks(thresholds)
plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
save_plot('6_combined_metrics_high_contrast.png')
