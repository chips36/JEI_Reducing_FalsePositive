import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 1. 최적 임계값(0.9) 기준 데이터 설정
# 여성 데이터 (GT: Female) -> TP: 293, FN: 7
# 남성 데이터 (GT: Male)   -> TN: 269, FP: 21

# 행렬 구성: [[TP, FN], [FP, TN]] 형태 (일반적인 논문 형식)
# 또는 [[True Female/Pred Female, True Female/Pred Male], [True Male/Pred Female, True Male/Pred Male]]

def plot_confusion_matrix(data, title, filename, labels=['Female', 'Male']):
    plt.figure(figsize=(7, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Female', 'Pred Male'],
                yticklabels=['Actual Female', 'Actual Male'],
                annot_kws={"size": 16, "weight": "bold"})

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"저장 완료: {filename}")
    plt.show()


# --- 1. 여성 데이터 전용 혼동 행렬 ---
# (실제 여성 300장 중 분류 결과)
female_cm = np.array([[293, 7], [0, 0]])  # 남성 데이터가 유입되지 않은 상태의 테스트
plot_confusion_matrix(female_cm, 'Confusion Matrix (Female Dataset Only)', 'cm_female.png')

# --- 2. 남성 데이터 전용 혼동 행렬 ---cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# (실제 남성 290장 중 분류 결과)
male_cm = np.array([[0, 0], [21, 269]])  # 여성 데이터가 유입되지 않은 상태의 테스트
plot_confusion_matrix(male_cm, 'Confusion Matrix (Male Dataset Only)', 'cm_male.png')

# --- 3. 통합 혼동 행렬 (가장 중요) ---
# 전체 590장 데이터를 한꺼번에 평가했을 때
combined_cm = np.array([[293, 7],  # 실제 여성 -> (TP, FN)
                        [21, 269]])  # 실제 남성 -> (FP, TN)
plot_confusion_matrix(combined_cm, 'Total Combined Confusion Matrix (Threshold 0.9)', 'cm_combined.png')