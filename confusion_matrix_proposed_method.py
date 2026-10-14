import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 스타일 설정
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

def plot_confusion_matrix(data, title, filename):
    plt.figure(figsize=(7, 6))
    # 선명한 파란색 계열(Blues)로 시각화
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Female', 'Pred Male'],
                yticklabels=['Actual Female', 'Actual Male'],
                annot_kws={"size": 18, "weight": "bold"})

    # 타이틀에 제안하는 방법 강조
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 저장 완료: {filename}")
    plt.show()

# =============================================================
# 1. YOLO 데이터 (Threshold 0.9 기준 - Proposed Method)
# =============================================================
yolo_combined_cm = np.array([
    [293, 7],   # Actual Female
    [21, 269]   # Actual Male
])

# =============================================================
# 2. NUDENET 데이터 (Threshold 0.9 기준 - Proposed Method)
# =============================================================
nudenet_combined_cm = np.array([
    [254, 28],  # Actual Female
    [18, 250]   # Actual Male
])

# --- 그래프 생성 ---
# 타이틀에 'Proposed Method'와 '제안하는 방법'을 병기하였습니다.
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