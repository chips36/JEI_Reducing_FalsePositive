import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot style
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

def plot_confusion_matrix(data, title, filename):
    """Draw a 2x2 confusion matrix heatmap for a baseline detector and save it (Fig. 3)."""
    plt.figure(figsize=(7, 6))
    # Baselines use a grey colormap to distinguish them from the proposed method (blue)
    sns.heatmap(data, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=['Pred Female', 'Pred Male'],
                yticklabels=['Actual Female', 'Actual Male'],
                annot_kws={"size": 18, "weight": "bold"})

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 저장 완료: {filename}")
    plt.show()

# =============================================================
# 1. YOLO baseline (object detection only)
# Female(300): TP=299, FN=1 | Male(300): TN=143, FP=147, Miss=10
# =============================================================
yolo_baseline_cm = np.array([
    [299, 1],    # Actual Female
    [147, 143]   # Actual Male (FP=147, TN=143 / 10 missed detections excluded)
])

# =============================================================
# 2. NudeNet baseline (object detection only)
# Female(300): TP=279, FN=3, Miss=18 | Male(300): TN=133, FP=135, Miss=32
# =============================================================
nudenet_baseline_cm = np.array([
    [279, 3],    # Actual Female (18 missed detections excluded)
    [135, 133]   # Actual Male (FP=135, TN=133 / 32 missed detections excluded)
])

# --- Generate the figures ---
print("🚀 YOLO Baseline 혼동 행렬 생성 중...")
plot_confusion_matrix(yolo_baseline_cm, 'YOLO Baseline (OD Only) Confusion Matrix', 'cm_yolo_baseline.png')

print("\n🚀 NudeNet Baseline 혼동 행렬 생성 중...")
plot_confusion_matrix(nudenet_baseline_cm, 'NudeNet Baseline (OD Only) Confusion Matrix', 'cm_nudenet_baseline.png')
