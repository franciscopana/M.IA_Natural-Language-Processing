import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _save_confusion_matrix(cm, class_names, filepath):
    plt.figure(figsize=(10, 8), facecolor='#eef1f5') 
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=class_names, 
                     yticklabels=class_names)

    ax.set_facecolor('#eef1f5') 
    plt.title('Confusion Matrix (1st assignment)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filepath, facecolor='#eef1f5') 
    plt.close()
    print(f"Confusion matrix saved to {filepath}")

confusion_matrix = [
    [2504, 15, 8, 5],
    [36, 2262, 177, 78],
    [23, 200, 2083, 242],
    [17, 115, 259, 2176]
]

label2id = {
    "impolite": 0,
    "neutral": 1,
    "polite": 2,
    "somewhat polite": 3
}

desired_order = ["impolite", "neutral", "somewhat polite", "polite"]
desired_indices = [label2id[label] for label in desired_order]

cm_np = np.array(confusion_matrix)
cm_reordered = cm_np[desired_indices, :][:, desired_indices]

_save_confusion_matrix(cm_reordered, desired_order, "confusion_matrix_reordered_1st_assign.png")
