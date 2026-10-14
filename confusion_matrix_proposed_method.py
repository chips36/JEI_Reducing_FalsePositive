import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot style
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

def plot_confusion_matrix(data, title, filename):
    """Draw a 2x2 confusion matrix heatmap for the proposed method and save it (Fig. 3)."""
    plt.figure(figsize=(7, 6))
    # Blue colormap for the proposed method
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Female', 'Pred Male'],
                yticklabels=['Actual Female', 'Actual Male'],
                annot_kws={"size": 18, "weight": "bold"})

    # The title highlights the proposed method
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 저장 완료: {filename}")
    plt.show()

# =============================================================
# 1. YOLO data (proposed method, threshold 0.9)
# =============================================================
yolo_combined_cm = np.array([
    [293, 7],   # Actual Female
    [21, 269]   # Actual Male
])

# =============================================================
# 2. NudeNet data (proposed method, threshold 0.9)
# =============================================================
nudenet_combined_cm = np.array([
    [254, 28],  # Actual Female
    [18, 250]   # Actual Male
])

# --- Generate the figures ---
print("🚀 YOLO (Proposed Method) 혼동 행렬 생성 중...")
plot_confusion_matrix(
    yolo_combined_cm,
    'YOLO: Proposed Method\n(Threshold 0.9)',
    'cm_yolo_proposed.png'
)

print("\n🚀 NudeNet (Proposed Method) 혼동 행렬 생성 중...")
plot_confusion_matrix(
    nudenet_combined_cm,
    'NudeNet: Proposed Method\n(Threshold 0.9)',
    'cm_nudenet_proposed.png'
)
