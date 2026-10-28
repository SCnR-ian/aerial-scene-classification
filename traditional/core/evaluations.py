from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_model_score(y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("\nModel evaluation results: ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    return accuracy, f1

def get_model_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nConfusion Matrix: ")
    print(cm)
    print("\nClassification Report: ")
    print(report)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()

#混淆矩阵可以换成热力图。 分析热力图，对于哪些类分类的不好
#在report中，可以单独的对某个指标进行分析（哪些类好，哪些差）


def plot_model_comparison(score_dict):

    models = list(score_dict.keys())
    accs = [score_dict[m][0] for m in models]
    f1s  = [score_dict[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x, accs, width=width, label='Accuracy')
    plt.bar(x + width, f1s, width=width, label='F1 Score')

    plt.xticks(x + width/2, models)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Performance comparison of different classifiers")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_robustness_bar_chart(modes, accuracies, f1_scores, title="Robustness test: perturbation comparison histogram"):
    """
    Draw a bar chart comparing Accuracy and F1-score under different perturbations
    Args:
    modes (List[str]): perturbation mode name, such as ["original", "noise", "blur", "occlusion"]
    accuracies (List[float]): corresponding Accuracy value
    f1_scores (List[float]): corresponding F1-score value
    title (str): chart title

    """
    x = range(len(modes))

    plt.figure(figsize=(8, 4))
    plt.bar(x, accuracies, width=0.4, label='Accuracy', align='center')
    plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1-score', align='center')

    plt.xticks([i + 0.2 for i in x], modes)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()