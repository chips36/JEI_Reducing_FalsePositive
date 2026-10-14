import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 1. Data at the optimal threshold (0.9), YOLOv8 + proposed method
# Female data (GT: Female) -> TP: 293, FN: 7
# Male data   (GT: Male)   -> TN: 269, FP: 21

# Matrix layout: [[TP, FN], [FP, TN]] (common paper format)
# i.e. [[True Female/Pred Female, True Female/Pred Male], [True Male/Pred Female, True Male/Pred Male]]

def plot_confusion_matrix(data, title, filename, labels=['Female', 'Male']):
    """Draw a 2x2 confusion matrix heatmap and save it as an image."""
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


# --- 1. Confusion matrix for the female dataset only ---
# (classification results of the 300 actual female images)
female_cm = np.array([[293, 7], [0, 0]])  # Test without any male data
plot_confusion_matrix(female_cm, 'Confusion Matrix (Female Dataset Only)', 'cm_female.png')

# --- 2. Confusion matrix for the male dataset only ---
# (classification results of the 290 detected male images)
male_cm = np.array([[0, 0], [21, 269]])  # Test without any female data
plot_confusion_matrix(male_cm, 'Confusion Matrix (Male Dataset Only)', 'cm_male.png')

# --- 3. Combined confusion matrix (most important) ---
# All 590 images evaluated together
combined_cm = np.array([[293, 7],  # Actual female -> (TP, FN)
                        [21, 269]])  # Actual male -> (FP, TN)
plot_confusion_matrix(combined_cm, 'Total Combined Confusion Matrix (Threshold 0.9)', 'cm_combined.png')
