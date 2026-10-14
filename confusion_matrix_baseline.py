import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 스타일 설정
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

def plot_confusion_matrix(data, title, filename):
    plt.figure(figsize=(7, 6))
    # Baseline은 구분을 위해 붉은색 계열(Reds) 또는 다른 색상을 권장하지만,
    # 비교를 위해 같은 파란색(Blues)으로 통일하거나 보라색(Purples)을 사용해도 좋습니다.
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
# 1. YOLO Baseline (Object Detection Only)
# Female(300): TP=299, FN=1 | Male(300): TN=143, FP=147, Miss=10
# =============================================================
yolo_baseline_cm = np.array([
    [299, 1],    # Actual Female
    [147, 143]   # Actual Male (FP=147, TN=143 / Miss 10 제외)
])

# =============================================================
# 2. NudeNet Baseline (Object Detection Only)
# Female(300): TP=279, FN=3, Miss=18 | Male(300): TN=133, FP=135, Miss=32
# =============================================================
nudenet_baseline_cm = np.array([
    [279, 3],    # Actual Female (Miss 18 제외)
    [135, 133]   # Actual Male (FP=135, TN=133 / Miss 32 제외)
])

# --- 그래프 생성 ---
print("🚀 YOLO Baseline 혼동 행렬 생성 중...")
plot_confusion_matrix(yolo_baseline_cm, 'YOLO Baseline (OD Only) Confusion Matrix', 'cm_yolo_baseline.png')

print("\n🚀 NudeNet Baseline 혼동 행렬 생성 중...")
plot_confusion_matrix(nudenet_baseline_cm, 'NudeNet Baseline (OD Only) Confusion Matrix', 'cm_nudenet_baseline.png')